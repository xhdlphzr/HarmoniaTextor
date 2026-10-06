# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Independent reviewer AI ("check AI") for composer sessions.

Every call to :meth:`ReviewerAI.review` opens a brand-new conversation, so the
reviewer never carries state between verdicts.  When it rejects a score the
creator session receives the suggestions and keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from harmoniatextor.agent.compression import (
    DEFAULT_CONTEXT_WINDOW,
    compress_messages,
    content_text,
    ensure_tool_responses,
)
from harmoniatextor.checker.rules import BUILTIN_RULES
from harmoniatextor.i18n import translate

__all__ = ["ReviewResult", "ReviewerAI"]

_REVIEW_TURNS = 3

_SYSTEM_TEMPLATE = (
    "你是一位严格、独立的{style}音乐评审专家。你只负责评审,不修改乐谱。\n"
    "重要:{rules} 等机械乐理规则已由程序化符号层严格校验并保证通过,"
    "你**不要**再重复检查,也**不要**以这些规则为由打回;你只做艺术、风格与表达层面的判断。\n"
    "请从以下维度审阅作品:\n"
    "1. 创作要求:对照【本乐章创作要求(Step 1 创作规划,含用户目标)】逐条核对是否满足;"
    "不满足必须打回,并指出缺了哪一条、应当怎样补。\n"
    "2. {style}风格:是否符合该风格的主题发展、织体、和声与语气特征,"
    "而非机械拼凑或混入不相称的风格。\n"
    "3. 结构完整:主题是否得到充分发展,整体是否成形而非片段堆砌。\n"
    "4. 意境与情感:作品是否表达出统一、真挚的意境与情感,而非机械拼凑。\n"
    "5. 节奏与旋律:左右手(或各声部)的节奏不要过于一致;整体节奏不要过于整齐,"
    "但变化也不能突兀;旋律进行要自然流畅、不割裂。\n"
    "6. 可演奏性:节奏与写法是否符合所选乐器的实际演奏。\n"
    "作品可以是单乐章或多乐章:若多乐章更合适应指出,但不要强求;"
    "若作品为多乐章,务必检查是否有清晰的乐章划分(各乐章的起止、速度与角色),"
    "缺少乐章划分必须打回并说明应如何划分。\n"
    "完成评审后必须调用 submit_review 工具给出结论:passed 为是否通过;"
    "suggestions 在打回时必须逐条写明小节号、涉及声部(谁和谁)以及具体修改办法,"
    "不要只给笼统结论。"
)


def _system_text(style_name: str, rules: frozenset[str]) -> str:
    """Render the reviewer system prompt for a style.

    Args:
        style_name: Active style display name.
        rules: Rule identifiers enforced for the work.

    Returns:
        The system prompt text.
    """
    names = "、".join(
        translate("rule." + rule.rule_id, "zh")
        for rule in BUILTIN_RULES
        if rule.rule_id in rules
    )
    return _SYSTEM_TEMPLATE.format(style=style_name, rules=names)


class _ReviewParams(BaseModel):
    """Arguments of the reviewer's submit tool.

    Attributes:
        passed: Whether the score passes the review.
        suggestions: Concrete problems and fixes when the score is rejected.
    """

    model_config = ConfigDict(extra="forbid")

    passed: bool = Field(description="Whether the score passes the review.")
    suggestions: str = Field(
        default="",
        description="Concrete problems and fixes; required when passed is false.",
    )


def _submit_review(passed: bool, suggestions: str = "") -> str:
    """Record the reviewer verdict.

    Args:
        passed: Whether the score passes.
        suggestions: Concrete problems and fixes.

    Returns:
        A confirmation string.
    """
    return "已收到评审结论。"


_REVIEW_TOOL = StructuredTool.from_function(
    func=_submit_review,
    name="submit_review",
    description="提交评审结论:passed 表示是否通过,suggestions 给出问题与修改建议。",
    args_schema=_ReviewParams,
)


@dataclass(slots=True)
class ReviewResult:
    """Outcome of one reviewer session.

    Attributes:
        passed: Whether the score passes the review.
        suggestions: Concrete problems and fixes when rejected.
    """

    passed: bool
    suggestions: str = ""


class ReviewerAI:
    """An independent reviewer that opens a fresh session for every verdict.

    Attributes:
        chat_model: Model driving the review.
        context_window: Context window used for compression.
    """

    def __init__(
        self,
        chat_model: BaseChatModel,
        *,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
    ) -> None:
        """Initialise the reviewer.

        Args:
            chat_model: Model driving the review.
            context_window: Context window used for compression.
        """
        self.chat_model = chat_model
        self.context_window = context_window

    def review(
        self,
        *,
        goal: str,
        genre_name: str,
        style_name: str,
        rules: frozenset[str],
        score_xml: str,
        check_summary: str,
    ) -> ReviewResult:
        """Review a score in a fresh session.

        Args:
            goal: The requirement the movement must satisfy (the Step 1 plan
                prompt, which itself embeds the user's goal).
            genre_name: Display name of the active genre.
            style_name: Display name of the active style.
            rules: Rule identifiers enforced for the work.
            score_xml: The complete current MusicXML.
            check_summary: Human-readable symbolic check summary.

        Returns:
            The review verdict.
        """
        messages: list[BaseMessage] = [
            SystemMessage(
                content=f"{_system_text(style_name, rules)}\n当前体裁:{genre_name}。"
            ),
            HumanMessage(
                content=(
                    f"【本乐章创作要求(Step 1 创作规划,含用户目标)】\n{goal}\n\n"
                    f"【符号层结果(已由程序校验,无需你复查)】\n{check_summary}\n\n"
                    f"完整乐谱 MusicXML:\n{score_xml}"
                )
            ),
        ]
        bound = self.chat_model.bind_tools([_REVIEW_TOOL])
        for _ in range(_REVIEW_TURNS):
            compress_messages(
                self.chat_model,
                messages,
                context_window=self.context_window,
                artifact_label="当前完整 MusicXML",
                artifact_provider=lambda: score_xml,
            )
            ensure_tool_responses(messages)
            response = bound.invoke(messages)
            verdict = _extract(response)
            if verdict is not None:
                return verdict
            messages.append(response)
            calls = getattr(response, "tool_calls", None) or []
            invalid = getattr(response, "invalid_tool_calls", None) or []
            if calls or invalid:
                messages.extend(
                    ToolMessage(
                        content="请调用 submit_review 工具给出评审结论。",
                        tool_call_id=str(call.get("id", "")),
                    )
                    for call in calls
                )
                messages.extend(
                    ToolMessage(
                        content=(
                            "submit_review 的参数解析失败,请重新调用并给出合法 JSON:"
                            f"{call.get('error', '')}"
                        ),
                        tool_call_id=str(call.get("id", "")),
                    )
                    for call in invalid
                )
            else:
                messages.append(
                    HumanMessage(
                        content="请调用 submit_review 工具给出评审结论(passed 与 suggestions)。"
                    )
                )
        return ReviewResult(False, "检查AI未能给出评审结论。")


def _extract(response: BaseMessage) -> ReviewResult | None:
    """Extract a review verdict from a model response.

    Args:
        response: The model response.

    Returns:
        The verdict, or ``None`` when the model did not call the submit tool.
    """
    calls = getattr(response, "tool_calls", None) or []
    for call in calls:
        if str(call.get("name", "")) == "submit_review":
            args = call.get("args", {})
            return ReviewResult(
                passed=bool(args.get("passed", False)),
                suggestions=content_text(args.get("suggestions", "")),
            )
    return None
