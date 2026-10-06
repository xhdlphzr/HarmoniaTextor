# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""Prompt construction for the composer agent."""

from __future__ import annotations

from harmoniatextor.checker.rules import BUILTIN_RULES, RULE_CONSTRAINTS
from harmoniatextor.genres.base import Genre
from harmoniatextor.i18n import translate
from harmoniatextor.styles.base import StyleKit
from harmoniatextor.techniques.registry import TechniqueRegistry

#: The composer and reviewer prompts are written in Chinese, so their rule and
#: technique names are resolved in Chinese regardless of the interface language.
_PROMPT_LANGUAGE = "zh"

__all__ = [
    "ARCHITECT_INSTRUCTION",
    "ARCHITECT_SYSTEM",
    "STEP1_INSTRUCTION",
    "STEP2_INSTRUCTION",
    "architect_instruction",
    "architect_system",
    "system_prompt",
]

ARCHITECT_SYSTEM = (
    "你是 HarmoniaTextor 的作曲规划智能体。你只做 Step 1 的架构规划,不写任何音符。"
    "你先用 set_title 为作品取一个简洁、贴切的标题,再用 add_movement 规划乐章,"
    "最后用 set_movement_prompt 为每一个乐章写一份**极其详细、可直接执行**的 prompt。"
    "没有乐章模板,结构完全由你写进 prompt。请尽量详细:调性与调式、拍号、速度、"
    "整体与分段结构(含各段小节数)、主要主题与动机(音名/音程/节奏轮廓)、对位与技法安排、"
    "声部数量与音域、乐器、声部交换与织体变化、力度与情绪走向、以及各段如何衔接。"
    "**每个乐章的 prompt 必须自成一体、可独立执行**:不要写“引用第 N 乐章的主题/材料”这类跨乐章指代,"
    "而要把需要沿用的音高、音程、节奏型、动机片段直接原样写进本乐章的 prompt 里;"
    "作曲时每个乐章只会拿到它自己这一份 prompt,拿不到别的乐章的文字。"
    "不要写空泛套话:每一句都要能指导后续逐音作曲;prompt 越长越具体越好。"
)

ARCHITECT_INSTRUCTION = (
    "Step 1 · 架构规划。现在开始用工具规划整部作品:\n"
    "1. 先用 set_title 为作品取一个简洁、贴切的标题;若始终不取,"
    "系统会用体裁名作标题。\n"
    "2. 先判断单乐章还是多乐章;若多乐章更合适就规划多个乐章,不要为凑数硬拆。\n"
    "3. 用 add_movement 逐个新增乐章(不需要选模板),工具会返回乐章编号(1、2、3…)。\n"
    "4. 用 set_movement_prompt(movement, prompt) 给每一个乐章写一份尽量详细的 prompt。"
    "至少涵盖:\n"
    "   (a) 调性与调式、拍号、速度与节拍性格;\n"
    "   (b) 本乐章结构与各段小节数(如呈示/展开/再现、A/B/A 等);\n"
    "   (c) 主要主题与动机的具体轮廓(音名/音程/节奏型)及其发展方式;\n"
    "   (d) 对位与技法安排(使用所选风格可用的技法)及位置;\n"
    "   (e) 声部数量、乐器与音域,声部交换与主奏轮换的位置;\n"
    "   (f) 力度、情绪走向与段落衔接。\n"
    "5. 每个乐章的 prompt 必须**独立、自足**:不要用“引用第一乐章的主题”“延续上一乐章材料”"
    "这类跨乐章指代——把要沿用的音高、音程、节奏型、动机片段原样写进本乐章 prompt;"
    "每个乐章作曲时只会拿到它自己这一份 prompt,拿不到其他乐章的文字。\n"
    "6. 必须覆盖全部乐章;系统会检查,若还有乐章没有 prompt 会让你继续补。\n"
    "规划要具体、可执行;生成阶段会严格按每个乐章的 prompt 逐个乐章作曲。"
    "每个乐章的 prompt 越长越具体越好。"
)

_ROLE = (
    "你是 HarmoniaTextor 的作曲家智能体,专门创作{style}风格的音乐。"
    "你通过工具把每一个音乐决定落地为 MusicXML;系统会在你完成时统一做符号层校验。"
    "你一次只负责**一个乐章**,并且是在独立的会话中工作:你只会拿到本乐章自己的"
    "创作要求(要求里已把需要沿用的素材完整写出),看不到其他乐章的文字,"
    "因此不要假设存在任何默认声部或既有材料,一切都从空谱开始由你亲手建立。"
)

_PROTOCOL = (
    "工作协议(请严格遵守以下全部限制):\n"
    "0. **本乐章的乐谱初始是空的,没有任何声部**。你必须用 add_part 按本乐章 prompt "
    "写明的乐器逐个创建声部;add_part 与 submit_theme **都必须显式给出 instrument**"
    "(如 Piano / Violin / Flute / Cello),系统**不会**替你推断乐器。"
    "**prompt 没有要求的声部一律不要创建**"
    "(例如钢琴独奏就只建钢琴声部,绝不建女高/女低/男高/男低等无关声部)。\n"
    "1. 用 submit_theme 提交主题旋律,并在这里决定调式:在 key 参数给出绝对调式"
    '(大写为大调、小写为小调,如 "C" 是 C 大调、"a" 是 a 小调);'
    "同时指定声部槽位(voice)与该声部的乐器(instrument,如 Violin/Flute/Cello/Oboe)。"
    "一个乐章可以提交多个主题,每个主题都会分配一个递增编号。\n"
    "2. 声部用 add_part 增、remove_part 删(都按声部槽位);同一乐器可拥有多个声部"
    "(如 violin1 与 violin2 都是 Violin)。**声部数量与编制严格以本乐章 prompt 为准**;"
    "需要调整整个乐章速度时用 set_tempo(bpm,四分音符/分钟)。\n"
    "3. 用 technique_* 工具引用主题编号并给出参数,逐步构建乐曲;"
    "要尽量多用**本风格可用技法**(见上方技法列表)来发展旋律,不要只把主题写一遍;"
    "需要转调或改变调式时,使用可用的移调/调性转换技法。"
    "节奏不要过于单调,要有层次感:长短音结合、强弱拍错落、声部间节奏对比与疏密变化。\n"
    "4. 除结构工具 insert/delete 外,修改类工具(submit_theme / add_part / technique_* / "
    "edit)**只返回 OK 状态、主题编号与违规列表,不返回完整乐谱**,以免上下文爆炸;"
    "需要查看当前完整谱时调用 read() 读取一份完整 MusicXML。\n"
    "5. 你先把乐谱写出来;当你停下(不再调用工具)时,系统会对当前乐谱做符号层校验"
    "并把违规返回给你。收到违规后,先 read() 取回当前谱,再依据违规信息用 "
    "edit(measure, voice, musicxml) **按小节号+声部**替换有问题的部分,然后再次确认完成。\n"
    "6. edit(measure, voice, musicxml):measure 是 1 起的小节号,voice 是声部槽位,"
    "musicxml 是一个只含该声部该小节旋律的片段。**若不提供片段(留空),则清空该小节的"
    "音符与休止**,可用来删除多余材料;清空或替换都合法,不会被拒绝。\n"
    "7. 结构编辑(注意与清空不同):insert(measure, voice, musicxml) 会在第 "
    "measure 位**插入**一个全新小节,所有声部的小节同步后移,可选填入某个声部;"
    "delete(measure) 会**删除**第 measure 小节,所有声部的小节同步前移;"
    "这两个工具会返回修改后的完整谱。需要加长/缩短乐曲时用它们,"
    "edit 只清空内容、不会增减小节。\n"
    "8. 每次只做一件事;引用主题时必须使用已分配的主题编号。\n"
    "9. 表情与记号用 annotate(measure, voice, mark, value) 添加:mark 取 "
    "dynamic(力度,value 如 pp/mf/f/ff)、text(表情文字,value 如 dolce)、"
    "crescendo/diminuendo(渐强/渐弱)、accent/tenuto/staccato(重音/保持音/跳音)、"
    "slur(连音线)、pedal(踏板)、tempo(乐章中途变速,value 为 BPM)。"
    "请在合适位置运用这些记号,让音乐更有表情与呼吸。\n"
    "10. 基本成型后,用 edit 逐小节微调收尾。修改时请多用装饰音"
    "(倚音、经过音、辅助音、回音等),适当调整部分音高与节奏以免单调,"
    "并适当留出空白(休止)让音乐有呼吸感。\n"
    "11. 必须逐条满足上面的符号层硬性限制,否则会被打回返工。\n"
    "12. 通过校验后总结你的作曲意图,供人工品鉴。"
)

_MUSICXML_FORMAT = (
    "MusicXML 参数格式(重要:**所有 musicxml 参数都是片段,不是完整乐谱**):\n"
    "- 片段必须是一个可解析的 MusicXML 文档,基本形如 "
    '`<?xml version="1.0" encoding="UTF-8"?><score-partwise><part><measure><note>'
    "<pitch><step>C</step><octave>5</octave></pitch><duration>1</duration>"
    "</note>...</measure></part></score-partwise>`;"
    "系统只读取其中**第一个 part**的音符(音高与时值),其余结构会被忽略。\n"
    "- submit_theme.musicxml:只含主题的**单声部、单行旋律**(可跨若干小节);"
    "调性由 key 参数决定、目标声部由 voice 参数决定;"
    "片段里的调号/拍号/声部名一律忽略。\n"
    "- edit.musicxml:只含**目标声部在目标这一小节**里的旋律;系统用它替换该小节;"
    "留空(不传或空串)表示清空该小节。\n"
    "- insert.musicxml:只含**新小节、目标声部**的旋律;留空表示插入一个空小节。\n"
    "- 不要提交整份总谱,也不要在片段里放多个声部或多个小节(多余内容会被忽略)。\n"
    "- read() 返回的才是**完整乐谱**;不要把 read 的输出原样再回传给任何修改工具。"
)

STEP1_INSTRUCTION = (
    "Step 1 · 创作规划。现在先不要调用任何工具,只输出一份文字规划:\n"
    "0. 先判断单乐章还是多乐章更合适:若多乐章更好就规划多个乐章,"
    "否则保持单乐章,不要为凑数硬拆。\n"
    "1. 乐章结构:列出全部乐章/段落,给出各自长度、速度与角色。\n"
    "2. 曲式:说明每个乐章采用的曲式(如赋格、二部/三部、回旋曲式、奏鸣曲式等)。\n"
    "3. 全部乐器与声部槽位(voice):逐一说明各自角色与音域。\n"
    "4. 情感分布:为每个声部写出情感走向,并说明各段落的情绪变化。\n"
    "5. 声部交换位置:指出哪些小节进行声部交换、主奏轮换或声部进出。\n"
    "6. 对位段落:标出模仿、密接和应、卡农等对位密集的段落及其起止小节。\n"
    "7. 调性布局与主要技法安排。\n"
    "规划要具体、可执行;接下来 Step 2 会严格据此实际作曲。"
)

STEP2_INSTRUCTION = (
    "Step 2 · 开始作曲。请按照上面的规划,使用 submit_theme、add_part 与 "
    "technique_* 等工具逐步把乐谱真正写出来。"
)


def _style_line(style: StyleKit) -> str:
    """Render the active style as a prompt line.

    Args:
        style: The active style kit.

    Returns:
        A one-line description with the style brief.
    """
    return f"当前风格:{style.name}。{style.brief}".rstrip("。")


def architect_system(style: StyleKit) -> str:
    """Build the architect system prompt for a style.

    Args:
        style: Active style kit.

    Returns:
        The system prompt text.
    """
    return f"{ARCHITECT_SYSTEM}\n\n{_style_line(style)}"


def architect_instruction(style: StyleKit) -> str:
    """Build the architect instruction for a style.

    Args:
        style: Active style kit.

    Returns:
        The instruction text.
    """
    return f"{ARCHITECT_INSTRUCTION}\n\n{_style_line(style)}"


def system_prompt(
    genre: Genre,
    style: StyleKit,
    techniques: TechniqueRegistry,
    rules: frozenset[str],
) -> str:
    """Build the system prompt for a composition session.

    The technique list and hard rule list are limited to the active style.

    Args:
        genre: Active genre.
        style: Active style kit.
        techniques: Technique registry used to enumerate available tools.
        rules: Rule identifiers enforced for this work.

    Returns:
        The system prompt text.
    """
    technique_lines = [
        f"- technique_{technique.id}"
        f"({translate('technique.' + technique.id, _PROMPT_LANGUAGE)}):"
        f"{technique.summary}"
        for technique in techniques.all()
    ]
    rule_lines = [
        f"- {rule.rule_id}"
        f"({translate('rule.' + rule.rule_id, _PROMPT_LANGUAGE)}):"
        f"{RULE_CONSTRAINTS.get(rule.rule_id, '')}"
        for rule in BUILTIN_RULES
        if rule.rule_id in rules
    ]
    return "\n".join(
        [
            _ROLE.replace("{style}", style.name),
            "",
            _style_line(style),
            f"当前体裁:{genre.display_name}({genre.id})。",
            "",
            "可用技法:",
            *technique_lines,
            "",
            _MUSICXML_FORMAT,
            "",
            "符号层硬性限制(违反任意一条都会被判失败并打回,必须逐条遵守):",
            *rule_lines,
            "",
            _PROTOCOL,
        ]
    )
