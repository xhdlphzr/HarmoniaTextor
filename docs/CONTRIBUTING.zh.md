<!--
SPDX-FileCopyrightText: 2026 xhdlphzr
SPDX-License-Identifier: MIT
-->

# 为 HarmoniaTextor 做贡献

感谢你对 HarmoniaTextor 的关注。本文档定义每一位贡献者都必须遵守的工程原则。
英文版见 [`CONTRIBUTING.md`](../CONTRIBUTING.md)。

## 快速开始

```console
uv sync --group dev
uv run pytest
uv run ht                     # pywebview 桌面窗口
uv run flask --app app.app run
```

要求：**Python 3.14+** 与 [uv](https://docs.astral.sh/uv/)。Linux 桌面外壳需要
`--extra desktop`（Qt WebEngine 后端）。

## 核心原则

以下原则不可妥协，且由 CI 强制校验。

1. **确定性的符号核心。** 所有音乐决定都用确定性算法实现，只有 LLM 是非确定性的。
   检查逻辑必须可证明正确，并用黄金用例锁定。
2. **100% 测试覆盖率。** `uv run pytest` 已固定为
   `--cov=src --cov=app --cov-fail-under=100`。每发现一个新 bug，都必须先用一个失败的
   回归测试复现，并审视测试是否有考虑不周之处。
3. **统一重复接口。** 当两个工具或辅助函数共享参数时，抽取公共模型，而不是复制字段。
4. **静态检查全绿。** `ruff check .` 必须覆盖 `src/`、`app/` 与 `tests/`；优先运行
   `ruff check --fix --unsafe-fixes`，再手动修改。应修复问题而非抑制；仅当违规确实
   属于有意为之，才退而使用行内 `# noqa`。
5. **严格类型。** `mypy --strict .` 必须覆盖 `src/`、`app/` 与 `tests/`；缺少 stub 包时
   先运行 `mypy --install-types`。优先补齐类型标注；对无 stub 的第三方库，先安装对应的
   `types-*` 包，只有确实没有 stub 且无法标注时才使用行内 `# type: ignore[...]`。
6. **Google 风格文档字符串**，覆盖每个模块、类与函数，由 Ruff 的 `D` 规则配合
   Google 约定强制（`ruff check --select D .`）；不使用独立的 `pydocstyle` 工具。
7. **格式化。** 每次修改后运行 `ruff format .`；CI 校验
   `ruff format --check .`。
8. **删除死代码。** 移除不可达或冗余代码，保持逻辑简洁。
9. **功能完整。** 在软件职能范围内尽可能满足真实使用场景。
10. **界面美观。** 圆角、英文标题用 Alex Brush（本地内置于 `app/static/fonts/`，OFL-1.1），
    其他用 Noto Serif SC 并带系统回退；不请求 CDN。必须具备滑动/展开动画。
11. **可复现环境。** `uv lock --check` 必须通过；提交前运行 `uv lock` 与 `uv sync`，
    并定期执行 `uv lock --upgrade`。
12. **REUSE 合规。** `reuse lint` 必须通过。每个文件都带有
    `SPDX-FileCopyrightText`/`SPDX-License-Identifier` 组合；逐文件版权头使用
    `xhdlphzr`，而 `LICENSE` 与 `REUSE.toml` 保持 `HarmoniaTextor contributors`。
    `assets/` 下的图标使用 `CC-BY-NC-ND-4.0`。
13. **CI/CD 纪律。** CI 只在 Ubuntu 上运行一次完整门禁。CD 在 Windows、macOS 与
    Ubuntu 上分别构建 PyInstaller 产物，并以每卷 1 GiB 的分卷上传到 GitHub Release。

## 质量门禁（严格顺序）

```console
uv lock --check
uv run reuse lint
uv run pytest
uv run ruff check .
uv run mypy --strict .
uv run ruff format --check .
uv run ruff check --select D .
```

最后一步在格式化之后重新检查模块、类与函数的 Google 风格文档字符串。本地鼓励先运行
`ruff check --fix --unsafe-fixes`，但不会自动提交修改。

## 开发流程

1. 新建分支。
2. 任何 bug 或行为变更，先写失败的测试。
3. 用最小的改动让测试通过。
4. 运行上面的完整质量门禁。
5. 提交 PR，说明改动内容以及覆盖它的测试。

### 测试评审清单

- 每个分支是否都有正例与反例？
- 空值/`None`/越界输入是否覆盖？
- 断言是否检验行为而非实现细节？
- 状态机的所有转移是否都被执行？
- mock 是否足够克制，不会掩盖真实 bug？

## 架构规则

- 符号层（`src/harmoniatextor/techniques`、`checker`、`genres`）不得掺杂 Web 与 LLM 逻辑。
- 新增技法、检查规则与体裁一律通过注册表接入，绝不改动引擎。
- 参数用 Pydantic 模型校验，并同时作为 LangChain 工具、HTTP API 与前端表单的 JSON Schema。
- Flask 应用与桌面外壳共享同一个 `CompositionService`。

## 许可证

本项目使用 MIT 许可证。提交贡献即表示你同意你的贡献以同一许可证发布。发起 PR 前
请运行 `reuse lint`。

## 多语言

英文文档位于仓库根目录；中文翻译位于 `docs/`（如 `docs/README.zh.md` 与
`docs/CONTRIBUTING.zh.md`）。修改面向用户的文档时请同步维护两种语言。
