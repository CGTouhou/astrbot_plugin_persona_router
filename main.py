"""人格路由插件 (astrbot_plugin_persona_router)

一个插件完成「动态选人格 + 人格上下文隔离」，不再依赖 astrbot_plugin_DynamicPersona。

设计脉络
--------
1. 选人格：用轻量 LLM（Selector）读人格规则和用户消息，判断本轮该用哪个人格。
   结果按 cache_ttl 缓存，同一个会话连续几条消息复用同一个判断。

2. 落人格：不做 system_prompt 注入，而是把选中的人格绑定到「对话」上
   （conversation.persona_id），交给 AstrBot 自己的 build_main_agent 去拼：

       # Persona Instructions
       ...
       ## Skills
       ...
       When using tools: ...

   这样人格卡被替换掉，而 Skills 段、工具调用约定、Computer Use 提示等
   全部原样保留。DynamicPersona 的 inject_mode=replace 会把这些一起抹掉，
   这正是本项目要解决的问题。

3. 隔离上下文（isolate_context 开关）：
   - 打开：每个人格绑定一条独立的 conversation（persona_id = 人格名），
     切人格 = 切对话，历史跟着人格走。人格由 AstrBot 原生注入。
   - 关闭：所有人格共用一条「共享对话」（persona_id 留空），历史连续；
     对话不绑人格，AstrBot 回落到全局默认人格照常注入，本插件再在 on_llm_request 里：
        replace（默认）= 直接替换 system_prompt（只剩选中人格）
        prepend        = 选中人格 + "\n\n" + 原 system_prompt（两层人格，Skills 段保住）

主流程时序（astrbot/core/pipeline/process_stage/method/agent_sub_stages/internal.py）：

    226  OnWaitingLLMRequestEvent   <- 本插件：选人格 + 切对话 + 写 selected_provider
    239  acquire_lock
    265  build_main_agent           <- AstrBot 按 conversation.persona_id 解析人格
    307  OnLLMRequestEvent          <- 本插件：只补 model、打诊断日志

设计原则
--------
1. 任何异常都只记日志，不阻塞主流程；失败时退化为 AstrBot 原生行为。
2. 复用 AstrBot 原生多对话机制，不自己维护历史文件。
3. WebUI 里给会话手动绑定的人格（session_service_config.persona_id）优先级最高，
   一旦存在，本插件整体让路。
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger, sp
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools, register
from sqlalchemy import null

_SELECTOR_TIMEOUT_SECONDS = 8
_MAX_CACHE_SIZE = 1024

# Selector 上下文用的「内存时间线」。
# 不读 AstrBot 的对话历史：隔离模式下每个人格一条对话，历史是分叉的；
# 非隔离模式又会把共享历史复制进人格对话。只有插件自己按 umo 记一份流水，
# 才能做到跨人格连续、不重复、不解析几十万字符的 JSON。
_TIMELINE_MAX = 30
_TIMELINE_TEXT_LIMIT = 300
# 长期运行保护：最多保留多少个会话、多久没更新就丢、写盘防抖延迟
_TIMELINE_UMO_LIMIT = 256
_TIMELINE_TTL_SECONDS = 7 * 86400
_TIMELINE_SAVE_DELAY = 3.0
# AstrBot 会把 [MSG_ID:...] 和 <system_reminder>...</system_reminder> 拼进
# message_str，这些元信息对判断人格没有意义，进时间线前先剔掉。
_RE_MSG_ID = re.compile(r"\[MSG_ID:-?\d+\]")
_RE_SYSTEM_REMINDER = re.compile(r"<system_reminder>[\s\S]*?(?:</system_reminder>|$)")

# 路由状态：本插件自己写、自己读，不依赖其它插件
_ROUTER_STATE_KEY = "persona_router_state"

# 共享对话的标题标记（非隔离模式）。对话的 persona_id 留空，AstrBot 的
# resolve_selected_persona 会回落到全局默认人格，原生注入照常进行，本插件
# 再在它前面叠加或替换。
# 注意 "[%None]" 是另一个语义 —— 那个表示「连默认人格都不要」，本插件不用。
_SHARED_TITLE = "[共享]"

# 本插件是唯一的人格决策方，用默认优先级即可
_HOOK_PRIORITY = 0
_DIAG_PRIORITY = -200

_SELECTOR_SYSTEM_PROMPT = """
你是一个对话风格分析器。你的任务是根据用户最新发送的消息，从给定的人格列表中选择最合适的一个。

规则：
1. 仔细阅读每个人格的「人格描述」和「使用该人格的情况」。
2. 分析用户消息的语义、情感倾向和意图。
3. 将用户意图与每个人格的「使用该人格的情况」进行匹配，选择最合适的人格。
4. 只返回 JSON 格式的结果，不要有任何其他文字。

返回格式（严格 JSON）：
{{"persona_id": "选中的人格ID"}}

如果无法判断或所有场景均不匹配，选择列表中第一个人格作为默认值。

{extra_prompt}
"""

_SELECTOR_SYSTEM_PROMPT_WITH_DEFAULT = """
你是一个对话风格分析器。判断用户最新消息是否命中了下列「特殊人格」的使用场景。

规则：
1. 仔细阅读每个人格的「人格描述」和「使用该人格的情况」。
2. 只有当用户消息明确命中某个场景时，才返回该人格 ID。
3. 命不中、拿不准、只是普通提问或闲聊时，返回 null —— 系统会使用默认人格。

返回格式（严格 JSON）：
{{"persona_id": "选中的人格ID"}}

如果没有任何人格的场景命中：
{{"persona_id": null}}

{extra_prompt}
"""

_SELECTOR_USER_PROMPT_TMPL = """
可选人格列表：{persona_list}

{recent_block}用户消息：{user_message}

请判断并按要求返回 JSON。
"""


@dataclass(slots=True)
class PersonaRule:
    persona_id: str
    persona_desc: str
    scenario_desc: str
    provider_id: str
    enabled: bool = True
    trigger_keywords: list[str] = field(default_factory=list)
    duration_seconds: int = 0
    rule_isolate: str = "inherit"  # inherit / on / off
    rule_mode: str = "inherit"  # inherit / prepend / replace


@dataclass(slots=True)
class PersonaDecision:
    persona_id: str
    provider_id: str = ""
    model_name: str = ""
    source: str = "selector"
    isolate: bool | None = None  # None = 跟随全局 isolate_context
    mode: str = ""  # "" = 跟随全局 inject_mode


@dataclass(slots=True)
class CachedDecision:
    decision: PersonaDecision
    hit_count: int = 0


@register(
    "astrbot_plugin_persona_router",
    "CGTouhou",
    "按聊天内容自动切换人格，每个人格各自拥有独立的聊天记录。",
    "2.5.8",
)
class PersonaRouterPlugin(Star):
    """动态人格增强：自动选人格 + 人格上下文隔离。"""

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config or {}
        self._persona_cache: dict[str, CachedDecision] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # 关键词触发后的持续状态：umo -> {"persona_id", "expires_at", "rule"}
        self._activations: dict[str, dict[str, Any]] = {}
        # Selector 上下文时间线：umo -> {"items": [{"role", "text", "persona"}], "updated": ts}
        self._timeline: dict[str, dict[str, Any]] = {}
        self._timeline_save_task: asyncio.Task | None = None

    async def initialize(self):
        rules = self._get_active_rules()
        keyword_rules = sum(1 for r in rules if r.trigger_keywords)
        logger.info(
            "[PersonaRouter] loaded enabled=%s filter=%s/%d cache_ttl=%s rules=%s "
            "keyword_rules=%s default=%s isolate=%s mode=%s ctx=%s timeout=%.1fs "
            "refresh=%s selector=%s",
            self.config.get("enabled", True),
            self.config.get("session_filter_mode", "disabled"),
            len(self.config.get("session_filter_list") or []),
            self.config.get("cache_ttl", 0),
            len(rules),
            keyword_rules,
            self._default_persona_id() or "(rules[0])",
            self.config.get("isolate_context", True),
            self.config.get("inject_mode", "replace"),
            self.config.get("selector_context_limit", 3),
            self._selector_timeout_seconds(),
            "on" if self.config.get("refresh_on_repeat", False) else "off",
            "on" if self.config.get("selector_enabled", True) else "off(keyword-only)",
        )
        # 人格名写错是最常见的配错方式，启动时集中报一次，别等运行时逐条 warning
        missing: list[str] = []
        default_pid = self._default_persona_id()
        if default_pid and not self._persona_exists(default_pid):
            missing.append(f"default_persona_id={default_pid}")
        for rule in rules:
            if not self._persona_exists(rule.persona_id):
                missing.append(f"规则人格={rule.persona_id}")
        if missing:
            logger.warning(
                "[PersonaRouter] 以下人格在 AstrBot 里不存在，对应规则会退化: %s",
                "、".join(missing),
            )
        # 把上次的时间线读回来，重启后 Selector 不会从零开始
        await self._load_timeline()
        # 启动对账：把上次接管过、这次配置改成不再管辖的会话解绑，
        # 否则那些会话的人格会卡在插件最后写下的 persona_id 上。
        await self._reconcile_bindings()

    # ---------------- 主钩子：选人格 + 切对话 ----------------
    @filter.on_waiting_llm_request(priority=_HOOK_PRIORITY)
    async def on_waiting_llm(self, event: AstrMessageEvent):
        if not self._should_run(event):
            # 这个会话不归本插件管（被会话过滤排除，或插件整体关闭）。
            # 顺手把它当前对话上残留的人格绑定拔掉——否则插件上次接管时写的
            # persona_id 会一直压住全局人格。拔掉之后该覆盖的由下次接管重写，
            # 不该覆盖的就纯走全局。
            await self._release_umo_binding(event.unified_msg_origin)
            return
        umo = event.unified_msg_origin
        try:
            if await self._has_forced_persona_binding(event):
                return
            rules = self._get_active_rules()
            default_pid = self._default_persona_id()
            # 没有规则可判；只有一条规则又没有默认人格时也判定不出什么
            if not rules:
                return
            if not default_pid and len(rules) < 2:
                return
            decision = await self._get_or_select_decision(event, rules, default_pid)
            if decision is None:
                return
            isolate = self._resolve_isolate(decision)
            self._apply_decision_to_event(event, decision, isolate)
            await self._switch_to_persona_conversation(event, decision, isolate)
        finally:
            # 放在 finally 里：Selector 判断那一刻时间线中还没有当前这句
            # （避免和 user_prompt 里的原文重复），而任何提前 return 的
            # 分支也照样把用户这句话记进来，时间线才不会缺口气。
            self._record_timeline(umo, "user", event.message_str)

    # ---------------- LLM 请求阶段：补 model + 诊断 ----------------
    @filter.on_llm_request(priority=_DIAG_PRIORITY)
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        state = event.get_extra(_ROUTER_STATE_KEY)
        injected = "native"
        layers = 0
        if isinstance(state, dict):
            if state.get("model_name"):
                req.model = state["model_name"]
            # 非隔离模式：原生人格照常注入，我们只负责在前面叠加/覆盖
            if not state.get("isolated") and state.get("persona_id"):
                mode = state.get("mode") or self.config.get("inject_mode", "replace")
                layers = await self._inject_persona_prompt(event, req, state["persona_id"], mode)
                if layers:
                    injected = "plugin"
        if not self.config.get("diagnose_prompt", True):
            return
        try:
            prompt = req.system_prompt or ""
            logger.info(
                "[PersonaRouter] prompt len=%d persona_section=%s skills=%s "
                "tool_call=%s injected=%s layers=%d source=%s",
                len(prompt),
                "# Persona Instructions" in prompt,
                "## Skills" in prompt,
                "never return an empty response" in prompt,
                injected,
                layers,
                (state or {}).get("source", "-") if isinstance(state, dict) else "-",
            )
        except Exception as exc:
            logger.warning("[PersonaRouter] prompt 诊断失败: %s", exc)

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, response: Any):
        """
        把机器人这一句记进内存时间线，供下一轮 Selector 判断对话走向。
        本插件的 Selector 调用走 context.llm_generate()，不经过
        MainAgentHooks.on_agent_done，所以不会被这个钩子捕获、不会污染时间线。
        """
        # 不归本插件管的会话不记时间线，省内存也省得污染判断
        if not self._should_run(event):
            return
        text = (getattr(response, "completion_text", "") or "").strip()
        if not text:
            return
        state = event.get_extra(_ROUTER_STATE_KEY)
        persona_id = ""
        if isinstance(state, dict):
            persona_id = str(state.get("persona_id") or "")
        self._record_timeline(event.unified_msg_origin, "assistant", text, persona_id)

    @filter.after_message_sent()
    async def on_after_message_sent(self, event: AstrMessageEvent):
        """
        会话被重置时，把本插件的内存态一起清掉。
        AstrBot 的 /reset 只清「当前对话」的历史。隔离模式下每个 (会话 × 人格)
        是一条独立对话，所以其它人格的记忆会原样保留；本插件的 _timeline /
        _activations / _persona_cache 更是完全不受影响。
        用户按 /reset 的直觉是「这个会话重开」，所以这里补全：
        - /reset：清空该会话下所有人格对话的历史，再清三个内存态
        - /new：只清三个内存态（旧对话历史要保留，那是 /new 的设计）
        """
        if not event.get_extra("_clean_group_context_session", False):
            return
        umo = event.unified_msg_origin
        raw = (getattr(event, "message_str", "") or "").strip().lower()
        is_reset = "reset" in raw
        is_new = (not is_reset) and ("new" in raw)
        cleared: list[str] = []
        if self._timeline.pop(umo, None) is not None:
            cleared.append("时间线")
        if self._activations.pop(umo, None) is not None:
            cleared.append("关键词激活")
        if self._persona_cache.pop(umo, None) is not None:
            cleared.append("Selector 缓存")
        if not (is_reset or is_new):
            logger.warning(
                "[PersonaRouter] 检测到会话重置标记但命令名无法识别（%r），只清内存态",
                raw,
            )
            return
        wiped = 0
        if is_reset and self.config.get("reset_all_personas", True):
            try:
                conversations = await self.context.conversation_manager.get_conversations(umo)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("[PersonaRouter] 读取会话对话列表失败: %s", exc)
                conversations = []
            for conv in conversations:
                if (conv.history or "").strip() in ("", "[]"):
                    continue
                try:
                    await self.context.conversation_manager.update_conversation(
                        umo, conv.cid, history=[]
                    )
                    wiped += 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error(
                        "[PersonaRouter] 清空对话 %s 历史失败: %s",
                        str(conv.cid)[:8],
                        exc,
                    )
        if is_reset:
            logger.info(
                "[PersonaRouter] /reset 全量重置 umo=%s：内存态[%s]；清空 %d 条人格对话历史",
                umo,
                "、".join(cleared) or "无",
                wiped,
            )
        else:
            logger.info(
                "[PersonaRouter] /new 新建对话 umo=%s：内存态[%s]",
                umo,
                "、".join(cleared) or "无",
            )

    async def _inject_persona_prompt(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
        persona_id: str,
        mode: str = "replace",
    ) -> int:
        """
        非隔离模式下把人格卡写进 system_prompt。
        返回实际叠了几层：
        - 0：没注入（人格不存在 / 原生已经注入了同一个 / 空卡）
        - 1：只有一层（replace，或原生没有人格时的 prepend）
        - 2：两层（原生默认人格 + 选中人格）
        """
        try:
            persona = self.context.persona_manager.get_persona_v3_by_id(persona_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[PersonaRouter] 读取人格 '%s' 失败，跳过注入: %s", persona_id, exc)
            return 0
        if not isinstance(persona, dict):
            logger.warning(
                "[PersonaRouter] 人格 '%s' 不存在，跳过注入（请检查人格是否被改名或删除）",
                persona_id,
            )
            return 0
        prompt = str(persona.get("prompt") or "").strip()
        if not prompt:
            return 0
        # 与 AstrBot 原生注入的排版保持一致，模型侧行为才会统一
        block = f"# Persona Instructions\n\n{prompt}"
        mode = str(mode or "replace").strip().lower()
        if mode == "replace":
            req.system_prompt = block
            layers = 1
        else:
            base = (req.system_prompt or "").strip()
            if not base:
                # 原生什么都没拼（没配全局人格），叠了个寂寞，算一层
                req.system_prompt = block
                layers = 1
            elif block in base:
                # 原生注入的已经是同一个人格，再叠就重复了。
                # 用完整 block 判断而不是裸 prompt：不同人格的 prompt 可能
                # 互为子串（比如都写了「你是一个助手」），裸 prompt 会误判成
                # 同一个人格而漏掉注入，那比重复叠加更糟。
                logger.debug("[PersonaRouter] 选中人格与原生人格相同，跳过叠加")
                layers = 0
            else:
                req.system_prompt = f"{block}\n\n{base}"
                layers = 2
        if layers:
            begin_dialogs = persona.get("_begin_dialogs_processed") or []
            if begin_dialogs:
                try:
                    req.contexts[:0] = copy.deepcopy(list(begin_dialogs))
                except Exception as exc:
                    logger.warning("[PersonaRouter] 注入开场白失败: %s", exc)
        return layers

    # ---------------- 选人格 ----------------
    def _should_run(self, event: AstrMessageEvent) -> bool:
        if not self.config.get("enabled", True):
            return False
        if not self._check_session_filter(event):
            return False
        if not (event.message_str or "").strip() and (
            not event.message_obj or not event.message_obj.message
        ):
            return False
        return True

    def _default_persona_id(self) -> str:
        return str(self.config.get("default_persona_id", "") or "").strip()

    def _persona_exists(self, persona_id: str) -> bool:
        """人格是否还在 AstrBot 里。查询本身异常时按「存在」处理，避免误报。"""
        try:
            persona = self.context.persona_manager.get_persona_v3_by_id(persona_id)
        except Exception:
            return True
        return isinstance(persona, dict)

    def _get_active_rules(self) -> list[PersonaRule]:
        raw_rules: list[dict[str, Any]] = self.config.get("persona_rules", []) or []
        rules: list[PersonaRule] = []
        for item in raw_rules:
            if not isinstance(item, dict):
                continue
            if not item.get("rule_enabled", True):
                continue
            persona_id = str(item.get("persona_id", "")).strip()
            if not persona_id:
                continue
            raw_keywords = item.get("trigger_keywords") or []
            if isinstance(raw_keywords, str):
                raw_keywords = [raw_keywords]
            keywords = [str(k).strip() for k in raw_keywords if str(k).strip()]
            try:
                duration = max(int(item.get("duration_seconds", 0) or 0), 0)
            except (TypeError, ValueError):
                duration = 0
            rule_isolate = str(item.get("rule_isolate", "inherit") or "inherit").strip().lower()
            if rule_isolate not in ("inherit", "on", "off"):
                rule_isolate = "inherit"
            rule_mode = str(item.get("rule_mode", "inherit") or "inherit").strip().lower()
            if rule_mode not in ("inherit", "prepend", "replace"):
                rule_mode = "inherit"
            rules.append(
                PersonaRule(
                    persona_id=persona_id,
                    persona_desc=str(item.get("persona_desc", "")).strip(),
                    scenario_desc=str(item.get("scenario_desc", "")).strip(),
                    provider_id=str(item.get("provider_id", "")).strip(),
                    trigger_keywords=keywords,
                    duration_seconds=duration,
                    rule_isolate=rule_isolate,
                    rule_mode=rule_mode,
                )
            )
        return rules

    def _resolve_isolate(self, decision: PersonaDecision) -> bool:
        """规则级 isolate 优先，未指定则跟随全局开关。"""
        if decision.isolate is not None:
            return decision.isolate
        return bool(self.config.get("isolate_context", True))

    def _resolve_mode(self, decision: PersonaDecision) -> str:
        """规则级注入方式优先，未指定则跟随全局。"""
        if decision.mode in ("prepend", "replace"):
            return decision.mode
        return str(self.config.get("inject_mode", "replace") or "replace").strip().lower()

    def _get_activation(self, umo: str) -> dict[str, Any] | None:
        """取当前未过期的关键词触发状态，过期则清掉。"""
        act = self._activations.get(umo)
        if not act:
            return None
        if float(act.get("expires_at", 0)) <= time.monotonic():
            self._activations.pop(umo, None)
            logger.info(
                "[PersonaRouter] 关键词人格到期，恢复默认 persona=%s",
                act.get("persona_id"),
            )
            return None
        return act

    def _set_activation(self, umo: str, rule: PersonaRule) -> None:
        """
        记录关键词触发的激活状态。
        重复命中同一个人格时默认不续期（原到期时间不动），只有换成另一个人格
        才立刻覆盖。想回到「每命中一次就重置计时」的老行为，把
        refresh_on_repeat 打开即可。
        """
        now = time.monotonic()
        if rule.duration_seconds <= 0:
            stale = self._activations.pop(umo, None)
            if stale:
                logger.info(
                    "[PersonaRouter] 关键词切回 persona=%s，清除原激活 persona=%s",
                    rule.persona_id,
                    stale.get("persona_id"),
                )
            return
        current = self._activations.get(umo)
        current_pid = str((current or {}).get("persona_id") or "")
        still_alive = bool(current) and float(current.get("expires_at", 0)) > now
        same_persona = bool(current) and current_pid == rule.persona_id
        if same_persona and still_alive:
            if not self.config.get("refresh_on_repeat", False):
                logger.info(
                    "[PersonaRouter] 关键词重复命中 persona=%s，保持原到期时间不续期",
                    rule.persona_id,
                )
                return
            self._activations[umo] = {
                "persona_id": rule.persona_id,
                "expires_at": now + float(rule.duration_seconds),
                "rule": rule,
            }
            logger.info(
                "[PersonaRouter] 关键词重复命中 persona=%s，已续期 %d 秒",
                rule.persona_id,
                rule.duration_seconds,
            )
            return
        if still_alive and not same_persona:
            logger.info(
                "[PersonaRouter] 关键词覆盖激活 persona=%s -> %s",
                current_pid,
                rule.persona_id,
            )
        self._activations[umo] = {
            "persona_id": rule.persona_id,
            "expires_at": now + float(rule.duration_seconds),
            "rule": rule,
        }
        logger.info(
            "[PersonaRouter] 关键词人格激活 persona=%s 持续 %d 秒",
            rule.persona_id,
            rule.duration_seconds,
        )

    def _match_keyword_rule(
        self, event: AstrMessageEvent, rules: list[PersonaRule]
    ) -> tuple[PersonaRule | None, int]:
        """按规则顺序做关键词/正则匹配，返回命中的规则。"""
        text = event.message_str or ""
        if not text.strip():
            return None, -1
        for idx, rule in enumerate(rules):
            for raw in rule.trigger_keywords:
                keyword = str(raw or "").strip()
                if not keyword:
                    continue
                try:
                    if re.search(keyword, text):
                        return rule, idx
                except re.error:
                    # 用户填的不是合法正则，退化成子串匹配
                    if keyword in text:
                        return rule, idx
        return None, -1

    async def _has_forced_persona_binding(self, event: AstrMessageEvent) -> bool:
        """WebUI 里手动给会话绑定人格时，让整个动态流程让路。"""
        try:
            session_service_config = (
                await sp.get_async(
                    scope="umo",
                    scope_id=str(event.unified_msg_origin),
                    key="session_service_config",
                    default={},
                )
                or {}
            )
            persona_id = str(session_service_config.get("persona_id", "")).strip()
            if persona_id and persona_id != "[%None]":
                logger.debug(
                    "[PersonaRouter] session %s forced to persona %s, skip",
                    event.unified_msg_origin,
                    persona_id,
                )
                return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[PersonaRouter] 检查强制人格失败，继续动态流程: %s", exc)
        return False

    async def _get_or_select_decision(
        self,
        event: AstrMessageEvent,
        rules: list[PersonaRule],
        default_pid: str,
    ) -> PersonaDecision | None:
        umo = event.unified_msg_origin
        # 1. 关键词命中：最高优先级，命中就覆盖已有的激活状态
        hit_rule, _ = self._match_keyword_rule(event, rules)
        if hit_rule is not None:
            # 新关键词代表一次明确的状态指令，Selector 缓存必须作废；否则
            # duration=0 的规则只影响这一条，下一条会命中缓存里的旧人格，
            # 而不是回到默认人格。续期与否交给 _set_activation 判断。
            self._persona_cache.pop(umo, None)
            self._set_activation(umo, hit_rule)
            logger.info("[PersonaRouter] 关键词命中 persona=%s", hit_rule.persona_id)
            return await self._build_decision(
                event, hit_rule.persona_id, rules, "keyword", hit_rule
            )
        # 2. 没有新关键词命中，看有没有还在激活期的人格
        activation = self._get_activation(umo)
        if activation:
            remain = max(int(float(activation.get("expires_at", 0)) - time.monotonic()), 0)
            logger.info(
                "[PersonaRouter] 关键词激活有效，跳过 LLM 判断 persona=%s 剩余 %d 秒",
                activation.get("persona_id"),
                remain,
            )
            return await self._build_decision(
                event,
                str(activation.get("persona_id") or ""),
                rules,
                "keyword-active",
                activation.get("rule"),
            )
        # 3. 纯关键词模式：不调 LLM、不用缓存，直接落到默认人格
        if not self.config.get("selector_enabled", True):
            fallback_id = default_pid or (rules[0].persona_id if rules else "")
            if not fallback_id:
                return None
            return await self._build_decision(event, fallback_id, rules, "keyword-only-default")
        # 4. 缓存
        cache_ttl = max(int(self.config.get("cache_ttl", 0) or 0), 0)
        cached = self._persona_cache.get(umo)
        if cached and cache_ttl > 0 and cached.hit_count < cache_ttl:
            cached.hit_count += 1
            return PersonaDecision(
                persona_id=cached.decision.persona_id,
                provider_id=cached.decision.provider_id,
                model_name=cached.decision.model_name,
                source="cache",
                isolate=cached.decision.isolate,
                mode=cached.decision.mode,
            )
        if cached is not None:
            self._persona_cache.pop(umo, None)
        # 5. Selector（LLM 判断）
        selected_persona_id, selector_status = await self._run_selector(event, rules, default_pid)
        if selected_persona_id:
            decision_source = "selector"
        elif selector_status == "null":
            # 模型明确判断「不匹配任何人格」——这是正常结果，不是失败。
            selected_persona_id = default_pid or (rules[0].persona_id if rules else "")
            decision_source = "selector-default"
        else:
            # 超时 / 调用失败 / 输出解析不了，才叫兜底。
            selected_persona_id = default_pid or (rules[0].persona_id if rules else "")
            decision_source = f"fallback-{selector_status}"
        if not selected_persona_id:
            return None
        decision = await self._build_decision(event, selected_persona_id, rules, decision_source)
        if cache_ttl > 0:
            self._persona_cache[umo] = CachedDecision(decision=decision, hit_count=1)
            self._prune_cache()
        return decision

    async def _build_decision(
        self,
        event: AstrMessageEvent,
        persona_id: str,
        rules: list[PersonaRule],
        source: str,
        rule: PersonaRule | None = None,
    ) -> PersonaDecision:
        provider_id = ""
        model_name = ""
        if rule is None:
            rule = next((r for r in rules if r.persona_id == persona_id), None)
        if rule is not None and rule.provider_id:
            provider = self.context.get_provider_by_id(rule.provider_id)
            if provider is not None:
                provider_id = rule.provider_id
                model_name = getattr(provider, "get_model", lambda: "")() or ""
            else:
                logger.warning(
                    "[PersonaRouter] 规则配置的 provider '%s' 不存在，跟随会话 provider",
                    rule.provider_id,
                )
        if not provider_id:
            provider_id = await self._get_current_chat_provider_id(event)
            if provider_id:
                provider = self.context.get_provider_by_id(provider_id)
                model_name = getattr(provider, "get_model", lambda: "")() or ""
        isolate: bool | None = None
        if rule is not None:
            if rule.rule_isolate == "on":
                isolate = True
            elif rule.rule_isolate == "off":
                isolate = False
        mode = ""
        if rule is not None and rule.rule_mode in ("prepend", "replace"):
            mode = rule.rule_mode
        return PersonaDecision(
            persona_id=persona_id,
            provider_id=provider_id,
            model_name=model_name,
            source=source,
            isolate=isolate,
            mode=mode,
        )

    async def _get_current_chat_provider_id(self, event: AstrMessageEvent) -> str:
        try:
            return await self.context.get_current_chat_provider_id(umo=event.unified_msg_origin)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[PersonaRouter] 获取会话 provider 失败: %s", exc)
            return ""

    def _apply_decision_to_event(
        self, event: AstrMessageEvent, decision: PersonaDecision, isolated: bool
    ) -> None:
        if decision.provider_id:
            event.set_extra("selected_provider", decision.provider_id)
        if decision.model_name:
            event.set_extra("selected_model", decision.model_name)
        event.set_extra(
            _ROUTER_STATE_KEY,
            {
                "persona_id": decision.persona_id,
                "provider_id": decision.provider_id,
                "model_name": decision.model_name,
                "source": decision.source,
                "isolated": isolated,
                "mode": self._resolve_mode(decision),
            },
        )

    def _selector_timeout_seconds(self) -> float:
        """Selector 超时时间（秒），来自配置，夹在 1–120 之间。"""
        try:
            value = float(self.config.get("selector_timeout_seconds", _SELECTOR_TIMEOUT_SECONDS))
        except (TypeError, ValueError):
            return float(_SELECTOR_TIMEOUT_SECONDS)
        if value <= 0:
            return float(_SELECTOR_TIMEOUT_SECONDS)
        return max(1.0, min(value, 120.0))

    async def _run_selector(
        self,
        event: AstrMessageEvent,
        rules: list[PersonaRule],
        default_pid: str,
    ) -> tuple[str | None, str]:
        """
        调 Selector 让 LLM 选人格。
        返回 (persona_id, status)：
        - ("贴吧大肥鱼", "ok")  选出来了
        - (None, "null")
        模型明确说「不匹配任何人格」（正常结果）
        - (None, "timeout")     超时
        - (None, "error")       调用失败
        - (None, "invalid")     模型吐了内容但解析不出合法人格
        """
        selector_provider_id = str(self.config.get("selector_provider_id", "") or "").strip()
        if not selector_provider_id:
            selector_provider_id = await self._get_current_chat_provider_id(event)
        if not selector_provider_id:
            logger.warning("[PersonaRouter] 没有可用的 selector provider，走兜底人格")
            return None, "error"
        persona_list_lines = []
        for index, rule in enumerate(rules, start=1):
            persona_list_lines.append(
                f"{index}. 人格 ID: {rule.persona_id}\n"
                f"   人格描述: {rule.persona_desc or '（无人格描述）'}\n"
                f"   使用该人格的情况: {rule.scenario_desc or '（无场景描述）'}"
            )
        extra_prompt = str(self.config.get("selector_prompt_extra", "") or "").strip()
        extra_section = f"\n附加偏好：{extra_prompt}" if extra_prompt else ""
        if default_pid:
            system_prompt = _SELECTOR_SYSTEM_PROMPT_WITH_DEFAULT.format(extra_prompt=extra_section)
        else:
            system_prompt = _SELECTOR_SYSTEM_PROMPT.format(extra_prompt=extra_section)
        context_limit = max(int(self.config.get("selector_context_limit", 3) or 0), 0)
        recent_context = await self._get_recent_context(event, context_limit)
        recent_block = ""
        if recent_context:
            logger.info(
                "[PersonaRouter] selector 参考最近 %d 条对话",
                len(recent_context.splitlines()),
            )
            recent_block = (
                "最近对话（只用来判断攻击对象和对话走向，不是待判断的消息）：\n"
                f"{recent_context}\n\n"
            )
        user_prompt = _SELECTOR_USER_PROMPT_TMPL.format(
            persona_list="\n".join(persona_list_lines),
            recent_block=recent_block,
            user_message=(event.message_str or "").strip() or "（空消息）",
        )
        try:
            llm_resp = await asyncio.wait_for(
                self.context.llm_generate(
                    chat_provider_id=selector_provider_id,
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                ),
                timeout=self._selector_timeout_seconds(),
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[PersonaRouter] selector 超时（%.1fs），走兜底人格",
                self._selector_timeout_seconds(),
            )
            return None, "timeout"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[PersonaRouter] selector 调用失败: %s", exc)
            return None, "error"
        raw_text = (getattr(llm_resp, "completion_text", "") or "").strip()
        selected = self._parse_selector_response(raw_text, rules)
        if selected:
            logger.info("[PersonaRouter] selector chose persona=%s", selected)
            return selected, "ok"
        if self._is_null_persona_reply(raw_text):
            logger.info("[PersonaRouter] selector 判定不匹配任何人格，用默认")
            return None, "null"
        logger.warning("[PersonaRouter] selector 输出无法解析，raw=%r", raw_text)
        return None, "invalid"

    async def _get_recent_context(self, event: AstrMessageEvent, limit: int) -> str:
        """
        取最近若干条消息，给 Selector 当上下文。
        优先读插件自己维护的内存时间线：跨人格连续、不重复、不用解析
        AstrBot 那份可能有几十万字符的 history。刚重启时间线还是空的，
        这时用当前对话历史预热一次，之后都走时间线。
        """
        if limit <= 0:
            return ""
        umo = event.unified_msg_origin
        if umo not in self._timeline:
            await self._prime_timeline(event, umo)
            if umo not in self._timeline:
                # 全新会话没有历史可预热。放个空桶占位，表示「已经试过」，
                # 否则之后每一轮都会白白去读一次历史。
                self._timeline[umo] = {"items": [], "updated": 0.0}
        lines = self._timeline_lines(umo, limit)
        if lines:
            return "\n".join(lines)
        # 连历史都没有（全新会话），保持空：Selector 只看当前消息
        return ""

    async def _prime_timeline(self, event: AstrMessageEvent, umo: str) -> None:
        """用当前对话历史预热时间线，只在新会话/重启后触发一次。"""
        history = await self._read_history(event)
        if not history:
            return
        bucket: list[dict[str, str]] = []
        for item in history[-_TIMELINE_MAX:]:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            if role not in ("user", "assistant"):
                continue
            text = self._extract_message_text(item.get("content"))
            if not text:
                continue
            bucket.append({"role": role, "text": text, "persona": ""})
        if bucket:
            # 用存在性判断而不是直接赋值：_read_history 里有 await，会让出
            # 事件循环，这期间可能已经有新消息记进时间线了，直接赋值会冲掉。
            if umo not in self._timeline:
                self._timeline[umo] = {"items": bucket, "updated": time.time()}
                self._schedule_timeline_save()
                logger.info("[PersonaRouter] 时间线冷启动，载入 %d 条历史", len(bucket))

    async def _read_history(self, event: AstrMessageEvent) -> list[Any]:
        """读当前对话的历史列表；任何异常都返回空，不影响主流程。"""
        umo = event.unified_msg_origin
        try:
            cid = await self.context.conversation_manager.get_curr_conversation_id(umo)
            if not cid:
                return []
            conversation = await self.context.conversation_manager.get_conversation(umo, cid)
            if conversation is None or not conversation.history:
                return []
            history = json.loads(conversation.history)
            if not isinstance(history, list):
                return []
            return history
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[PersonaRouter] 读取历史失败: %s", exc)
            return []

    @staticmethod
    def _history_as_list(history: Any) -> list[Any]:
        """
        把 Conversation.history 统一解成 list。

        AstrBot 的 Conversation.history 是 JSON 字符串，而 update_conversation
        把传进来的 history 直接写进 JSON 列、不会先 dump 一次。把字符串原样
        传回去会变成双重编码：库里存的是 '"[]"'，下次 json.loads 一次拿到的
        还是 str。插件这边有 isinstance 兜底，会静默当成空历史；AstrBot 那边
        读到 str 直接丢进 req.contexts 则会报错。所以搬历史前一律过这里。
        """
        if isinstance(history, list):
            return history
        if not isinstance(history, str):
            return []
        text = history.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            return []
        if isinstance(parsed, list):
            return parsed
        # 兜底：库里已经存了双重编码的脏数据，再解一层
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except (TypeError, ValueError):
                return []
            if isinstance(parsed, list):
                return parsed
        return []

    def _record_timeline(self, umo: str, role: str, text: str, persona_id: str = "") -> None:
        """往内存时间线追加一条消息，超出上限自动丢最旧的。"""
        # 纯关键词模式不会调用 Selector，这份时间线没人读，别白占内存
        if not self.config.get("selector_enabled", True):
            return
        text = self._clean_text(text)
        if not text:
            return
        if len(text) > _TIMELINE_TEXT_LIMIT:
            text = text[:_TIMELINE_TEXT_LIMIT] + "…"
        bucket = self._timeline.get(umo)
        if bucket is None:
            bucket = {"items": [], "updated": 0.0}
            self._timeline[umo] = bucket
        items = bucket["items"]
        items.append({"role": role, "text": text, "persona": persona_id or ""})
        if len(items) > _TIMELINE_MAX:
            del items[: len(items) - _TIMELINE_MAX]
        bucket["updated"] = time.time()
        self._prune_timeline()
        self._schedule_timeline_save()

    def _timeline_lines(self, umo: str, limit: int) -> list[str]:
        """把时间线末尾 limit 条渲染成「用户: ...」形式，机器人带人格名。"""
        bucket = (self._timeline.get(umo) or {}).get("items") or []
        if not bucket:
            return []
        lines: list[str] = []
        for item in bucket[-limit:]:
            text = item.get("text") or ""
            if not text:
                continue
            if item.get("role") == "user":
                lines.append(f"用户: {text}")
            else:
                persona = item.get("persona") or ""
                label = f"机器人（{persona}）" if persona else "机器人"
                lines.append(f"{label}: {text}")
        return lines

    @staticmethod
    def _clean_text(text: str) -> str:
        """剔掉 AstrBot 拼进来的元信息，只留用户真正说的话。"""
        if not text:
            return ""
        cleaned = _RE_SYSTEM_REMINDER.sub(" ", text)
        cleaned = _RE_MSG_ID.sub(" ", cleaned)
        return " ".join(cleaned.split())

    # ---------------- 时间线持久化 ----------------

    def _timeline_file(self) -> Path:
        """时间线落盘位置：data/plugin_data/<插件名>/timeline.json。"""
        try:
            base = StarTools.get_data_dir("astrbot_plugin_persona_router")
        except Exception:
            base = Path("data") / "plugin_data" / "astrbot_plugin_persona_router"
        return Path(base) / "timeline.json"

    async def _load_timeline(self) -> None:
        """启动时把上次的时间线读回来，重启后 Selector 判断不会断档。"""
        path = self._timeline_file()
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[PersonaRouter] 读取时间线失败，忽略: %s", exc)
            return
        if not isinstance(raw, dict):
            return
        now = time.time()
        loaded = 0
        for umo, payload in raw.items():
            if not isinstance(payload, dict):
                continue
            items = payload.get("items")
            try:
                updated = float(payload.get("updated") or 0)
            except (TypeError, ValueError):
                updated = 0.0
            if not isinstance(items, list) or not items:
                continue
            if now - updated > _TIMELINE_TTL_SECONDS:
                continue
            clean: list[dict[str, str]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = self._clean_text(str(item.get("text") or ""))
                if not text:
                    continue
                if len(text) > _TIMELINE_TEXT_LIMIT:
                    text = text[:_TIMELINE_TEXT_LIMIT] + "…"
                clean.append(
                    {
                        "role": item.get("role") or "user",
                        "text": text,
                        "persona": item.get("persona") or "",
                    }
                )
            if not clean:
                continue
            self._timeline[str(umo)] = {
                "items": clean[-_TIMELINE_MAX:],
                "updated": updated or now,
            }
            loaded += 1
        self._prune_timeline()
        if loaded:
            logger.info("[PersonaRouter] 时间线已恢复 %d 个会话", loaded)

    def _prune_timeline(self) -> None:
        """丢掉过期会话；会话数超上限时丢最久没更新的。"""
        now = time.time()
        for umo, bucket in list(self._timeline.items()):
            if now - float(bucket.get("updated") or 0) > _TIMELINE_TTL_SECONDS:
                self._timeline.pop(umo, None)
        overflow = len(self._timeline) - _TIMELINE_UMO_LIMIT
        if overflow > 0:
            ordered = sorted(self._timeline.items(), key=lambda kv: float(kv[1].get("updated") or 0))
            for umo, _ in ordered[:overflow]:
                self._timeline.pop(umo, None)
        # 时间线里已经没了的会话，它的锁也一起收掉，避免长期运行后无限堆积
        for umo in [k for k in self._locks if k not in self._timeline]:
            self._locks.pop(umo, None)

    def _schedule_timeline_save(self, delay: float = _TIMELINE_SAVE_DELAY) -> None:
        """防抖写盘：短时间内的多次改动只写一次，避免每条消息都碰磁盘。"""
        if self._timeline_save_task and not self._timeline_save_task.done():
            return
        try:
            self._timeline_save_task = asyncio.create_task(self._save_timeline_later(delay))
        except RuntimeError:
            self._timeline_save_task = None

    async def _save_timeline_later(self, delay: float) -> None:
        await asyncio.sleep(delay)
        await self._save_timeline()

    async def _save_timeline(self) -> None:
        """原子写：先写临时文件再替换，避免写到一半留下坏文件。"""
        path = self._timeline_file()
        now = time.time()
        payload: dict[str, Any] = {}
        for umo, bucket in self._timeline.items():
            items = bucket.get("items") or []
            if not items:
                continue
            updated = float(bucket.get("updated") or now)
            if now - updated > _TIMELINE_TTL_SECONDS:
                continue
            payload[umo] = {"items": items[-_TIMELINE_MAX:], "updated": updated}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[PersonaRouter] 保存时间线失败: %s", exc)

    def _extract_message_text(self, content: Any) -> str:
        """从 history 的 content 里抽纯文本，兼容字符串和分段列表两种形态。"""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for segment in content:
                if isinstance(segment, str):
                    parts.append(segment)
                elif isinstance(segment, dict):
                    if segment.get("type") == "text" and segment.get("text"):
                        parts.append(str(segment["text"]))
            text = " ".join(parts)
        else:
            return ""
        text = self._clean_text(text)
        if len(text) > 300:
            text = text[:300] + "…"
        return text

    def _parse_selector_response(
        self,
        raw_text: str,
        rules: list[PersonaRule],
    ) -> str | None:
        valid_id_set = {rule.persona_id for rule in rules if rule.persona_id}
        candidate_texts = [raw_text]
        fragment = self._extract_json_object(raw_text)
        if fragment and fragment != raw_text:
            candidate_texts.append(fragment)
        for text in candidate_texts:
            try:
                data = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(data, dict):
                continue
            persona_id = data.get("persona_id")
            if persona_id is None:
                return None
            persona_id = str(persona_id).strip()
            if persona_id in valid_id_set:
                return persona_id
        return None

    def _is_null_persona_reply(self, raw_text: str) -> bool:
        """
        Selector 的回复是不是在说「不匹配任何人格」。
        配置了默认人格时，提示词要求命不中返回 {"persona_id": null}。
        这种情况是正常判断结果，不能和超时/解析失败混成一类。
        """
        if not raw_text:
            return False
        candidates = [raw_text]
        fragment = self._extract_json_object(raw_text)
        if fragment and fragment != raw_text:
            candidates.append(fragment)
        for text in candidates:
            try:
                data = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(data, dict):
                continue
            if "persona_id" not in data:
                continue
            value = data.get("persona_id")
            return value is None or str(value).strip().lower() in (
                "",
                "null",
                "none",
            )
        return raw_text.strip().lower() in ("null", "none", "{}")

    def _extract_json_object(self, text: str) -> str | None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]

    def _prune_cache(self) -> None:
        overflow = len(self._persona_cache) - _MAX_CACHE_SIZE
        if overflow <= 0:
            return
        for key in list(self._persona_cache)[:overflow]:
            self._persona_cache.pop(key, None)

    # ---------------- 人格对话隔离 ----------------
    async def _switch_to_persona_conversation(
        self, event: AstrMessageEvent, decision: PersonaDecision, isolate: bool
    ) -> None:
        persona_id = decision.persona_id
        umo = event.unified_msg_origin
        lock = self._locks.get(umo)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[umo] = lock
        try:
            async with lock:
                if isolate:
                    target = await self._ensure_persona_conversation(event, umo, persona_id)
                else:
                    target = await self._ensure_shared_conversation(event, umo)
                if target is None:
                    logger.warning(
                        "[PersonaRouter] 未能准备对话 persona=%s isolate=%s umo=%s",
                        persona_id,
                        isolate,
                        umo,
                    )
                    return
                curr_cid = await self.context.conversation_manager.get_curr_conversation_id(umo)
                switched = curr_cid != target.cid
                if switched:
                    await self.context.conversation_manager.switch_conversation(umo, target.cid)
                logger.info(
                    "[PersonaRouter] 路由人格 persona=%s cid=%s isolate=%s switched=%s",
                    persona_id,
                    str(target.cid)[:8],
                    isolate,
                    switched,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "[PersonaRouter] 人格上下文切换失败（已退化为原生行为）: %s",
                exc,
                exc_info=True,
            )

    @staticmethod
    def _conversation_belongs(conv: Any, persona_id: str) -> bool:
        """判断一条对话是否属于某个人格。

        两层锚：`persona_id` 是活绑定，`title == "[人格名]"` 是永久铭牌。
        插件退出管辖时会抹掉当前对话的 persona_id，但人格对话的 title 永远
        保留，这样插件重新接管、判回同一人格时还能把那段历史找回来。
        """
        if (conv.persona_id or "") == persona_id:
            return True
        return (conv.title or "").strip() == f"[{persona_id}]"

    async def _ensure_persona_conversation(
        self, event: AstrMessageEvent, umo: str, persona_id: str
    ):
        """找到该人格的对话；没有就认领一条旧对话或新建一条。"""
        conv_mgr = self.context.conversation_manager
        # 快路径：当前对话已经是目标人格，省掉一次全表查询
        curr_cid = await conv_mgr.get_curr_conversation_id(umo)
        if curr_cid:
            current = await conv_mgr.get_conversation(umo, curr_cid)
            if current is not None and self._conversation_belongs(current, persona_id):
                # /new 建出来的对话会继承当前人格的 persona_id（AstrBot 的
                # _get_current_persona_id 直接返回 conv.persona_id），但没有
                # title。补上铭牌，否则它在插件眼里是「主对话」，插件退场时
                # _release_umo_binding 会把它当候选，把会话指针拨来拨去。
                # 只补空对话：有历史的主对话要继续保持无标题，退场时靠
                # title 为空把它认出来，把指针拨回去。
                if not (current.title or "").strip() and not self._history_as_list(
                    current.history
                ):
                    try:
                        await conv_mgr.update_conversation(
                            umo, current.cid, title=f"[{persona_id}]"
                        )
                        current.title = f"[{persona_id}]"
                        logger.info(
                            "[PersonaRouter] 补齐对话铭牌 persona=%s cid=%s",
                            persona_id,
                            str(current.cid)[:8],
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.warning("[PersonaRouter] 补齐对话铭牌失败: %s", exc)
                return current
        conversations = await conv_mgr.get_conversations(umo)
        matched = [c for c in conversations if self._conversation_belongs(c, persona_id)]
        if matched:
            # 同一人格可能有多条（用户手动 /new 过），取最近更新的那条
            return max(matched, key=lambda c: c.updated_at or 0)
        if self.config.get("claim_legacy_conversation", True):
            # 只认领「完全无主」的对话：persona_id 为空、title 也为空。
            # title 非空的都是本插件建的（`[人格名]` / `[共享]`），它们的主人
            # 由 title 铭牌锁定，绝不能因为 persona_id 被解绑就认给别的人格，
            # 否则贴吧的历史会被嫁给优化后，等于人格串味。
            legacy = [
                c for c in conversations if not (c.persona_id or "") and not (c.title or "").strip()
            ]
            # 主对话（title 为空）是 AstrBot 原生建的那条，通常装着插件接管
            # 之前积累的全部历史，比插件自己建的人格对话值钱。但它只交给
            # 默认人格——否则群里第一次撞上的若是「对骂人格」，十几万字正常
            # 聊天记录就被它端走了，之后判回默认人格反而要新建空对话。
            default_pid = str(self.config.get("default_persona_id") or "").strip()
            candidate = None
            candidate_is_empty = False
            if legacy and (not default_pid or persona_id == default_pid):
                candidate = max(legacy, key=lambda c: c.updated_at or 0)
            if candidate is None and legacy:
                # 空的无主对话（用户刚发过 /new、或手动建了没聊天的对话）
                # 没有历史可丢，任何人格都能收编。否则本次会另建一条
                # `[人格名]`，这条空对话留在列表里当孤儿，插件退场时
                # _release_umo_binding 还会按 updated_at 把它选成主对话。
                empty_legacy = [c for c in legacy if not self._history_as_list(c.history)]
                if empty_legacy:
                    candidate = max(empty_legacy, key=lambda c: c.updated_at or 0)
                    candidate_is_empty = True
            if candidate is not None:
                # 空对话顺手补上铭牌：否则它 title 仍为空，插件退场时
                # _release_umo_binding 会把它当成「主对话」候选，把会话指针
                # 拨到一条刚认领的空对话上。
                new_title = f"[{persona_id}]" if candidate_is_empty else None
                await conv_mgr.update_conversation(
                    umo, candidate.cid, persona_id=persona_id, title=new_title
                )
                candidate.persona_id = persona_id
                if new_title:
                    candidate.title = new_title
                logger.info(
                    "[PersonaRouter] 认领旧对话 persona=%s cid=%s",
                    persona_id,
                    str(candidate.cid)[:8],
                )
                return candidate
        cid = await conv_mgr.new_conversation(
            umo,
            event.get_platform_id(),
            title=f"[{persona_id}]",
            persona_id=persona_id,
        )
        # 从共享对话（非隔离）切回隔离模式时，新的人格对话是空的，
        # 直接把当前历史搬过去，避免用户感觉「刚聊完就失忆」。
        if self.config.get("inherit_history", True) and curr_cid:
            try:
                current = await conv_mgr.get_conversation(umo, curr_cid)
                if current is not None and (current.title or "").strip() == _SHARED_TITLE:
                    inherited = self._history_as_list(current.history)
                    if inherited:
                        # 注意必须传 list：update_conversation 直接把 history
                        # 写进 JSON 列，传字符串会被二次编码。
                        await conv_mgr.update_conversation(umo, cid, history=inherited)
                        logger.info(
                            "[PersonaRouter] 人格对话继承共享历史 persona=%s cid=%s",
                            persona_id,
                            str(cid)[:8],
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[PersonaRouter] 继承共享历史失败: %s", exc)
        target = await conv_mgr.get_conversation(umo, cid)
        if target is None:
            logger.warning(
                "[PersonaRouter] 新建对话后读取失败 persona=%s cid=%s",
                persona_id,
                cid,
            )
            return None
        logger.info(
            "[PersonaRouter] 新建人格对话 persona=%s cid=%s",
            persona_id,
            str(target.cid)[:8],
        )
        return target

    async def _ensure_shared_conversation(self, event: AstrMessageEvent, umo: str):
        """
        非隔离模式：所有人格共用一条「不绑人格」的对话。
        对话的 persona_id 留空（NULL），AstrBot 的 resolve_selected_persona
        会回落到全局默认人格，原生注入照常拼进 system_prompt。
        本插件再按 inject_mode 处理：replace（默认）整段覆盖，只剩选中人格，
        Skills 段会一起丢掉；prepend 在它前面叠一层，形成「选中人格 + 默认人格」
        两层，Skills 段和工具约定跟着保留。
        """
        conv_mgr = self.context.conversation_manager
        curr_cid = await conv_mgr.get_curr_conversation_id(umo)
        current = None
        if curr_cid:
            current = await conv_mgr.get_conversation(umo, curr_cid)
            if current is not None and (current.title or "").strip() == _SHARED_TITLE:
                return current
            # 用户发了 /new：AstrBot 建的是无 title、无 persona 的空对话。
            # 如果直接回去找旧的 [共享]，这条空对话就变成孤儿，`/new` 等于没
            # 生效；而且插件退场时 _release_umo_binding 会按 updated_at 把它
            # 选中，用户以为历史被清了。空对话没有历史可丢，就地收编。
            if (
                current is not None
                and not (current.title or "").strip()
                and not (current.persona_id or "").strip()
                and not self._history_as_list(current.history)
            ):
                await conv_mgr.update_conversation(umo, current.cid, title=_SHARED_TITLE)
                current.title = _SHARED_TITLE
                logger.info(
                    "[PersonaRouter] 收编空对话为共享对话 cid=%s",
                    str(current.cid)[:8],
                )
                return current
        conversations = await conv_mgr.get_conversations(umo)
        # 只认 title 是 [共享] 的那条。不能按「persona_id 为空」来认——
        # 主对话和刚被解绑过的人格对话 persona_id 也是空，按空匹配 + 取最近
        # 更新的那条，会把某个人格的历史当成共享历史端走。
        shared = [c for c in conversations if (c.title or "").strip() == _SHARED_TITLE]
        if shared:
            return max(shared, key=lambda c: c.updated_at or 0)
        # 第一次进入非隔离模式：把当前对话的历史搬进共享对话，
        # 否则从隔离切过来的瞬间等于失忆。
        inherited_history: list[Any] = []
        if self.config.get("inherit_history", True) and curr_cid:
            try:
                if current is None:
                    current = await conv_mgr.get_conversation(umo, curr_cid)
                if current is not None:
                    inherited_history = self._history_as_list(current.history)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[PersonaRouter] 读取当前对话历史失败: %s", exc)
        cid = await conv_mgr.new_conversation(
            umo,
            event.get_platform_id(),
            title=_SHARED_TITLE,
        )
        if inherited_history:
            try:
                # 必须传 list 而不是 conversation.history 字符串：后者会被
                # update_conversation 二次编码进 JSON 列，历史就废了。
                await conv_mgr.update_conversation(umo, cid, history=inherited_history)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[PersonaRouter] 共享对话继承历史失败: %s", exc)
        target = await conv_mgr.get_conversation(umo, cid)
        if target is None:
            logger.warning("[PersonaRouter] 新建共享对话后读取失败 cid=%s", cid)
            return None
        logger.info(
            "[PersonaRouter] 新建共享对话 cid=%s 继承历史=%s",
            str(target.cid)[:8],
            bool(inherited_history),
        )
        return target

    # ---------------- 会话过滤 ----------------
    def _check_session_filter(self, event: AstrMessageEvent) -> bool:
        mode = str(self.config.get("session_filter_mode") or "disabled").strip()
        if mode == "disabled":
            return True
        raw_list = self.config.get("session_filter_list") or []
        targets = {str(x).strip() for x in raw_list if str(x).strip()}
        if not targets:
            return mode == "blacklist"
        umo = event.unified_msg_origin
        group_id = event.get_group_id() or ""
        sender_id = event.get_sender_id() or ""
        session_id = event.message_obj.session_id if event.message_obj else ""
        # 群聊里只认群号，不认发送者：否则黑名单里填一个 QQ 号，这个人
        # 在哪个群说话，哪个群就会被整群排除。发送者 ID 只在私聊参与匹配。
        if group_id:
            candidates = {group_id, session_id}
        else:
            candidates = {sender_id, session_id}
        candidates.discard("")
        hit = any(fid in candidates for fid in targets) or any(
            umo == t or umo.endswith(":" + t) for t in targets
        )
        return hit if mode == "whitelist" else not hit

    # ---------------- 人格绑定解绑 ----------------

    async def _unbind_persona(self, umo: str, cid: str) -> bool:
        """把一条对话的 persona_id 清成 NULL（等价于 AstrBot 的「回落全局人格」）。

        走 conversation_manager 的公开接口：persona_id 传 SQLAlchemy 的 null()
        表达式。它不是 None，能穿过 update_conversation 里
        `if persona_id is not None` 的拦截，到 ORM 层生成
        `SET persona_id = NULL`。结果与直接写库一致，但全程不碰数据库连接、
        不依赖内部属性，也没有 LIKE 模糊更新。
        """
        conv_mgr = self.context.conversation_manager
        try:
            await conv_mgr.update_conversation(umo, cid, persona_id=null())
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "[PersonaRouter] 解绑对话 %s 失败: %s", str(cid)[:8], exc
            )
            return False

    async def _release_persona_bindings(self) -> tuple[int, int]:
        """把所有会话的指针拨回主对话，并解掉主对话的人格绑定。

        返回 (处理条数, 扫描会话数)。人格专属对话一律不碰，它们的 title 就是
        `[人格名]`，插件重新接管时靠它把历史找回来。
        """
        conv_mgr = self.context.conversation_manager
        try:
            conversations = await conv_mgr.get_conversations()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("[PersonaRouter] 枚举会话失败: %s", exc, exc_info=True)
            return 0, 0

        umos = {conv.user_id for conv in conversations if (conv.user_id or "").strip()}
        released = 0
        for umo in umos:
            if await self._release_umo_binding(umo):
                released += 1
        return released, len(umos)

    def _session_allowed_by_umo(self, umo: str) -> bool:
        """只靠 umo 判断某个会话是否归本插件管（后台对账时手头没有 event）。

        规则和 _check_session_filter 一致，只少了 group_id / sender_id 那两位
        精确比对——umo 末段本来就是群号或用户 ID，用后缀匹配覆盖得到。
        """
        mode = str(self.config.get("session_filter_mode") or "disabled").strip()
        if mode == "disabled":
            return True
        targets = {
            str(x).strip() for x in (self.config.get("session_filter_list") or []) if str(x).strip()
        }
        if not targets:
            return mode == "blacklist"
        hit = any(umo == t or umo.endswith(":" + t) for t in targets)
        return hit if mode == "whitelist" else not hit

    async def _release_umo_binding(self, umo: str) -> bool:
        """让会话脱离插件管辖：指针拨回主对话，并解掉主对话的人格绑定。

        只动主对话（title 为空、AstrBot 原生创建的那条）。人格专属对话一律
        不碰——它们的 title 是 `[人格名]`，是「人格 ↔ 历史」的永久锚点，插件
        重新接管时靠它把对话找回来。解绑只抹 persona_id 一列，历史一个字不动。
        """
        conv_mgr = self.context.conversation_manager
        try:
            conversations = await conv_mgr.get_conversations(umo)
            if not conversations:
                return False

            mains = [c for c in conversations if not (c.title or "").strip()]
            if mains:
                target = max(mains, key=lambda c: c.updated_at or 0)
                changed = False
                curr_cid = await conv_mgr.get_curr_conversation_id(umo)
                if str(curr_cid or "") != str(target.cid):
                    await conv_mgr.switch_conversation(umo, target.cid)
                    changed = True
                    logger.info(
                        "[PersonaRouter] 会话不归本插件管，指针拨回主对话 umo=%s cid=%s",
                        umo,
                        str(target.cid)[:8],
                    )
                if (target.persona_id or "").strip():
                    if await self._unbind_persona(umo, str(target.cid)):
                        changed = True
                        logger.info(
                            "[PersonaRouter] 会话不归本插件管，解绑主对话 umo=%s cid=%s 原人格=%s",
                            umo,
                            str(target.cid)[:8],
                            target.persona_id,
                        )
                return changed

            # 没有主对话（历史全被插件建的人格对话占着）时，退化成解绑当前
            # 对话，至少保证人格不会被钉死。
            cid = await conv_mgr.get_curr_conversation_id(umo)
            if not cid:
                return False
            conv = await conv_mgr.get_conversation(umo, cid)
            if not conv or not (conv.persona_id or "").strip():
                return False
            if await self._unbind_persona(umo, str(cid)):
                logger.info(
                    "[PersonaRouter] 会话不归本插件管，解绑当前对话 umo=%s cid=%s 原人格=%s",
                    umo,
                    str(cid)[:8],
                    conv.persona_id,
                )
                return True
            return False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("[PersonaRouter] 释放 %s 失败: %s", umo, exc)
            return False

    async def _reconcile_bindings(self) -> None:
        """启动对账：把不归本插件管的会话上残留的人格绑定全部拔掉。

        覆盖「上次运行时还在管、这次配置改成不管了」的情况。运行时改动过滤
        配置则由 on_waiting_llm 的兜底分支逐会话感知，不依赖这里。
        """
        conv_mgr = self.context.conversation_manager
        try:
            conversations = await conv_mgr.get_conversations()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("[PersonaRouter] 启动对账：枚举会话失败: %s", exc, exc_info=True)
            return
        umos = {conv.user_id for conv in conversations if (conv.user_id or "").strip()}
        released = 0
        for umo in umos:
            if self._session_allowed_by_umo(umo):
                continue
            if await self._release_umo_binding(umo):
                released += 1
        if released:
            logger.info(
                "[PersonaRouter] 启动对账：%d 个不受管辖的会话已解绑人格绑定",
                released,
            )

    # ---------------- 管理命令 ----------------

    @filter.command_group("dp")
    def dp(self):
        """人格路由管理"""
        return None

    @dp.command("status")  # type: ignore[attr-defined]
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def dp_status(self, event: AstrMessageEvent):
        rules = self._get_active_rules()
        cache_ttl = max(int(self.config.get("cache_ttl", 0) or 0), 0)
        cache = self._persona_cache.get(event.unified_msg_origin)
        if cache is not None:
            cache_text = (
                f"{cache.decision.persona_id} / "
                f"provider={cache.decision.provider_id or 'session-default'} / "
                f"model={cache.decision.model_name or 'provider-default'} / "
                f"hits={cache.hit_count}/{cache_ttl}"
            )
        else:
            cache_text = "无"
        curr_cid = await self.context.conversation_manager.get_curr_conversation_id(
            event.unified_msg_origin
        )
        curr_persona = "(无)"
        if curr_cid:
            conv = await self.context.conversation_manager.get_conversation(
                event.unified_msg_origin, curr_cid
            )
            if conv is not None:
                curr_persona = conv.persona_id or "(未绑定)"
        activation = self._get_activation(event.unified_msg_origin)
        if activation:
            remain = max(int(float(activation.get("expires_at", 0)) - time.monotonic()), 0)
            activation_text = f"{activation.get('persona_id')}（还剩 {remain} 秒）"
        else:
            activation_text = "无"
        timeline_bucket = self._timeline.get(event.unified_msg_origin) or {}
        timeline_count = len(timeline_bucket.get("items") or [])
        lines = [
            "人格路由状态",
            f"enabled: {self.config.get('enabled', True)}",
            f"default_persona_id: {self._default_persona_id() or '(rules[0])'}",
            f"active_rules: {len(rules)}",
            f"isolate_context: {self.config.get('isolate_context', True)}",
            f"inject_mode: {self.config.get('inject_mode', 'replace') or 'replace'}（仅非隔离时生效）",
            f"inherit_history: {self.config.get('inherit_history', True)}",
            f"selector_enabled: {self.config.get('selector_enabled', True)}"
            + ("（纯关键词模式）" if not self.config.get("selector_enabled", True) else ""),
            f"selector_provider_id: {self.config.get('selector_provider_id') or 'follow-session'}",
            f"selector_context_limit: {self.config.get('selector_context_limit', 3)}",
            f"selector_timeout: {self._selector_timeout_seconds():.1f}s",
            f"时间线: {timeline_count} 条（跨人格，落盘）",
            f"cache_ttl: {cache_ttl}",
            f"关键词激活: {activation_text}",
            f"cache: {cache_text}",
            f"当前对话: {str(curr_cid)[:8] if curr_cid else '无'} / persona={curr_persona}",
        ]
        for rule in rules:
            keyword_hint = (
                f" / 关键词{len(rule.trigger_keywords)}条" if rule.trigger_keywords else ""
            )
            duration_hint = f" / 持续{rule.duration_seconds}秒" if rule.duration_seconds > 0 else ""
            isolate_hint = (
                f" / isolate={rule.rule_isolate}" if rule.rule_isolate != "inherit" else ""
            )
            mode_hint = f" / mode={rule.rule_mode}" if rule.rule_mode != "inherit" else ""
            lines.append(
                f"- {rule.persona_id}: provider={rule.provider_id or 'follow-session'}"
                f"{keyword_hint}{duration_hint}{isolate_hint}{mode_hint}"
            )
        yield event.plain_result("\n".join(lines))

    @dp.command("now")  # type: ignore[attr-defined]
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def dp_now(self, event: AstrMessageEvent):
        """一句话看当前生效的人格。用法：/dp now"""
        umo = event.unified_msg_origin
        activation = self._get_activation(umo)
        if activation:
            remain = max(int(float(activation.get("expires_at", 0)) - time.monotonic()), 0)
            yield event.plain_result(
                f"当前人格：{activation.get('persona_id')}" f"（关键词激活中，还剩 {remain} 秒）"
            )
            return
        curr_cid = await self.context.conversation_manager.get_curr_conversation_id(umo)
        curr_persona = "(未绑定)"
        if curr_cid:
            conv = await self.context.conversation_manager.get_conversation(umo, curr_cid)
            if conv is not None:
                curr_persona = conv.persona_id or "(未绑定)"
        cache_ttl = max(int(self.config.get("cache_ttl", 0) or 0), 0)
        cache = self._persona_cache.get(umo)
        lines = ["关键词激活：无"]
        if cache is not None:
            lines.append(
                f"上一条用的：{cache.decision.persona_id}"
                f"（Selector 缓存 {cache.hit_count}/{cache_ttl}，下一条可能变化）"
            )
        elif not self.config.get("selector_enabled", True):
            lines.append(f"默认人格：{self._default_persona_id() or '(rules[0])'}")
        else:
            lines.append(
                f"默认人格：{self._default_persona_id() or '(rules[0])'}"
                "（还没有判断记录，下一条消息会重新判断）"
            )
        lines.append(
            f"当前对话：{str(curr_cid)[:8] if curr_cid else '无'} / persona={curr_persona}"
        )
        yield event.plain_result("\n".join(lines))

    @dp.command("clear")  # type: ignore[attr-defined]
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def dp_clear(self, event: AstrMessageEvent):
        """清掉当前会话的关键词激活状态和 Selector 时间线。用法：/dp clear"""
        umo = event.unified_msg_origin
        removed = self._activations.pop(umo, None)
        cleared = self._timeline.pop(umo, None)
        timeline_count = len((cleared or {}).get("items") or [])
        if cleared:
            await self._save_timeline()
        parts = []
        if removed:
            parts.append(f"关键词激活（原人格：{removed.get('persona_id')}）")
        if timeline_count:
            parts.append(f"Selector 时间线（{timeline_count} 条）")
        if parts:
            yield event.plain_result("已清除" + "、".join(parts) + "，下一条消息会重新判断。")
        else:
            yield event.plain_result("当前会话没有关键词激活状态，时间线也是空的。")

    @dp.command("refresh")  # type: ignore[attr-defined]
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def dp_refresh(self, event: AstrMessageEvent):
        count = len(self._persona_cache)
        self._persona_cache.clear()
        yield event.plain_result(f"已清空判断缓存（{count} 个会话），下一条消息会重新判断。")

    @dp.command("enable")  # type: ignore[attr-defined]
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def dp_enable(self, event: AstrMessageEvent):
        self.config["enabled"] = True
        self.config.save_config()
        yield event.plain_result("人格路由已启用。")

    @dp.command("disable")  # type: ignore[attr-defined]
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def dp_disable(self, event: AstrMessageEvent):
        self.config["enabled"] = False
        self.config.save_config()
        yield event.plain_result("人格路由已禁用。")

    @dp.command("migrate")  # type: ignore[attr-defined]
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def dp_migrate(self, event: AstrMessageEvent, persona_id: str = ""):
        """把当前对话绑定到指定人格。用法：/dp migrate 优化后"""
        persona_id = (persona_id or "").strip()
        if not persona_id:
            yield event.plain_result("用法：/dp migrate <人格名>，例如 /dp migrate 优化后")
            return
        umo = event.unified_msg_origin
        cid = await self.context.conversation_manager.get_curr_conversation_id(umo)
        if not cid:
            yield event.plain_result("当前会话还没有对话，先发一条消息再说。")
            return
        try:
            await self.context.conversation_manager.update_conversation(
                umo, cid, persona_id=persona_id
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("[PersonaRouter] migrate 失败: %s", exc, exc_info=True)
            yield event.plain_result(f"绑定失败：{exc}")
            return
        yield event.plain_result(f"已把当前对话 {str(cid)[:8]} 绑定到人格「{persona_id}」。")

    @dp.command("restore")  # type: ignore[attr-defined]
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def dp_restore(self, event: AstrMessageEvent):
        """解绑所有人格对话绑定，回落到全局人格。

        用途：停用插件、切换到 whitelist、或者杀进程之后，把残留的
        conversation.persona_id 清掉。用法：/dp restore
        """
        released, total = await self._release_persona_bindings()
        if total == 0:
            yield event.plain_result("没有找到任何会话，无需解绑。")
            return
        yield event.plain_result(
            f"扫描 {total} 个会话，解绑 {released} 条对话的人格绑定。\n"
            "这些会话的人格现在回落到全局设置；历史记录一条没动。\n"
            "本插件之后若重新启用，这些对话会被当作旧对话重新认领。"
        )

    @dp.command("sessionid")  # type: ignore[attr-defined]
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def dp_sessionid(self, event: AstrMessageEvent):
        lines = [
            f"group_id: {event.get_group_id() or 'private'}",
            f"sender_id: {event.get_sender_id() or 'unknown'}",
            f"session_id: {event.message_obj.session_id if event.message_obj else 'unknown'}",
            f"umo: {event.unified_msg_origin}",
        ]
        yield event.plain_result("\n".join(lines))

    async def terminate(self):
        """插件停用 / 重载 / AstrBot 正常退出时调用。

        把残留的人格对话绑定解绑，避免插件退场后会话卡在最后选中的人格上。
        覆盖不到的情况：强制杀进程、whitelist 排除——这两种要用 /dp restore。
        """
        self._persona_cache.clear()
        # 取消还没到点的延迟写盘任务，避免退出时留下 pending task
        if self._timeline_save_task and not self._timeline_save_task.done():
            self._timeline_save_task.cancel()
            try:
                await self._timeline_save_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self._timeline_save_task = None
        try:
            await self._save_timeline()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[PersonaRouter] terminate 保存时间线失败: %s", exc)
        try:
            released, total = await self._release_persona_bindings()
            logger.info(
                "[PersonaRouter] terminate：扫描 %d 个会话，解绑 %d 条人格绑定",
                total,
                released,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("[PersonaRouter] terminate 解绑失败: %s", exc, exc_info=True)
