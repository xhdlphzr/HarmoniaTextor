/*
SPDX-FileCopyrightText: 2026 xhdlphzr
SPDX-License-Identifier: MIT
*/

(function () {
  "use strict";

  function byId(id) {
    return document.getElementById(id);
  }

  function on(id, event, handler) {
    const element = byId(id);
    if (element) {
      element.addEventListener(event, handler);
    }
  }

  function setHref(id, url) {
    const element = byId(id);
    if (element) {
      element.href = url;
    }
  }

  function showToast(message, kind) {
    let container = byId("toast-container");
    if (!container) {
      container = document.createElement("div");
      container.id = "toast-container";
      container.className = "toast-container";
      document.body.appendChild(container);
    }
    const toast = document.createElement("div");
    toast.className = "toast" + (kind ? ` ${kind}` : "");
    toast.textContent = message;
    container.appendChild(toast);
    window.setTimeout(() => {
      toast.classList.add("hide");
      window.setTimeout(() => toast.remove(), 320);
    }, 2600);
  }

  function copyScore(url) {
    fetch(url)
      .then((response) => response.text())
      .then((xml) => {
        if (!xml || !xml.trim()) {
          showToast("尚无乐谱可复制。", "bad");
          return;
        }
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(xml).then(
            () => showToast("已复制 MusicXML。", "ok"),
            () => showToast("复制失败，请手动选择文字复制。", "bad")
          );
        } else {
          showToast("复制失败，请手动选择文字复制。", "bad");
        }
      })
      .catch(() => showToast("读取乐谱失败。", "bad"));
  }

  document.addEventListener("click", (event) => {
    const target = event.target;
    const link = target && target.closest ? target.closest("a.export-download") : null;
    if (!link) {
      return;
    }
    event.preventDefault();
    const base = link.href.split("?")[0];
    showToast("正在导出…", "ok");
    fetch(`${base}?save=1`)
      .then((response) => response.json())
      .then((data) => {
        if (data.ok) {
          showToast(`导出成功：${data.path || ""}`, "ok");
        } else {
          showToast(data.error || "导出失败。", "bad");
        }
      })
      .catch(() => showToast("导出失败。", "bad"));
  });

  function autoGrow(textarea) {
    const maxLines = Number(textarea.dataset.maxLines || 13);
    const style = window.getComputedStyle(textarea);
    const lineHeight = parseFloat(style.lineHeight) || 20;
    const padding = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
    const border = parseFloat(style.borderTopWidth) + parseFloat(style.borderBottomWidth);
    const maxHeight = lineHeight * maxLines + padding + border;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, maxHeight)}px`;
    textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
  }

  function bindAutoGrow(root) {
    (root || document).querySelectorAll("textarea.autogrow").forEach((textarea) => {
      autoGrow(textarea);
      textarea.addEventListener("input", () => autoGrow(textarea));
    });
  }

  bindAutoGrow(document);

  /* ----------------------------------------------------------- staff render */

  const osmdState = new WeakMap();

  function renderScoreInto(container, xml) {
    if (!container) {
      return Promise.resolve();
    }
    container.classList.remove("hidden");
    const text = xml ? xml.replace(/^\uFEFF/, "").trim() : "";
    if (!text) {
      container.textContent = "尚无乐谱。";
      return Promise.resolve();
    }
    if (text.indexOf("<?xml") !== 0) {
      container.textContent = "乐谱数据异常，请刷新后重试。";
      if (window.console) {
        window.console.error("Invalid MusicXML payload:", text.slice(0, 160));
      }
      return Promise.resolve();
    }
    if (typeof opensheetmusicdisplay === "undefined") {
      container.textContent = "未加载乐谱渲染组件（请强制刷新 Ctrl+F5）。";
      return Promise.resolve();
    }
    const state = osmdState.get(container) || { token: 0 };
    osmdState.set(container, state);
    state.token += 1;
    const token = state.token;
    container.innerHTML = "";
    const osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(container, {
      autoResize: false,
      backend: "svg",
      drawTitle: false,
      drawComposer: false,
    });
    return osmd
      .load(text)
      .then(() => {
        if (osmdState.get(container) === state && state.token === token) {
          osmd.render();
        }
      })
      .catch((error) => {
        if (osmdState.get(container) === state && state.token === token) {
          container.textContent = "乐谱渲染失败，请强制刷新（Ctrl+F5）后重试。";
        }
        if (window.console) {
          window.console.error("OSMD render failed:", error);
        }
      });
  }

  function collectFontFaces() {
    let css = "";
    Array.from(document.styleSheets).forEach((sheet) => {
      let rules;
      try {
        rules = sheet.cssRules;
      } catch (error) {
        return;
      }
      Array.from(rules).forEach((rule) => {
        if (rule.type === CSSRule.FONT_FACE_RULE) {
          css += rule.cssText + "\n";
        }
      });
    });
    return css;
  }

  function downloadScorePng(container, filename) {
    const svg = container.querySelector("svg");
    if (!svg) {
      window.alert("请先显示五线谱。");
      return;
    }
    const clone = svg.cloneNode(true);
    const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
    style.textContent = collectFontFaces();
    clone.insertBefore(style, clone.firstChild);
    const box = svg.viewBox && svg.viewBox.baseVal;
    const width = (box && box.width) || svg.getBoundingClientRect().width;
    const height = (box && box.height) || svg.getBoundingClientRect().height;
    clone.setAttribute("viewBox", `0 0 ${width} ${height}`);
    clone.setAttribute("width", String(width));
    clone.setAttribute("height", String(height));
    const data = new XMLSerializer().serializeToString(clone);
    const url = "data:image/svg+xml;base64," + window.btoa(unescape(encodeURIComponent(data)));
    const image = new Image();
    image.onload = () => {
      const scale = 2;
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(width * scale));
      canvas.height = Math.max(1, Math.round(height * scale));
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#fffdf8";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      const link = document.createElement("a");
      link.href = canvas.toDataURL("image/png");
      link.download = filename;
      link.click();
      showToast(`导出成功：${filename}`, "ok");
    };
    image.onerror = () => showToast("导出 PNG 失败。", "bad");
    image.src = url;
  }

  /* ---------------------------------------------------------------- config */

  function fieldValue(id) {
    const node = byId(id);
    return node ? node.value.trim() : "";
  }

  function setFieldValue(id, value) {
    const node = byId(id);
    if (node) {
      node.value = value || "";
    }
  }

  function openConfig() {
    fetch("/api/config")
      .then((response) => response.json())
      .then((config) => {
        setFieldValue("cfg-base-url", config.base_url);
        setFieldValue("cfg-api-key", config.api_key);
        setFieldValue("cfg-model", config.model);
        setFieldValue("cfg-context-window", config.context_window);
        const modal = byId("config-modal");
        if (modal) {
          modal.classList.remove("hidden");
        }
      })
      .catch(() => window.alert("读取配置失败。"));
  }

  function closeConfig() {
    const modal = byId("config-modal");
    if (modal) {
      modal.classList.add("hidden");
    }
  }

  function saveConfig() {
    const windowValue = Number(fieldValue("cfg-context-window"));
    const payload = {
      base_url: fieldValue("cfg-base-url"),
      api_key: fieldValue("cfg-api-key"),
      model: fieldValue("cfg-model"),
      context_window: windowValue > 0 ? windowValue : 200,
    };
    fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.ok) {
          closeConfig();
        } else {
          window.alert(data.error || "保存失败。");
        }
      })
      .catch(() => window.alert("保存失败。"));
  }

  const configBtn = byId("config-btn");
  if (configBtn) {
    configBtn.addEventListener("click", openConfig);
  }
  const configSave = byId("config-save");
  if (configSave) {
    configSave.addEventListener("click", saveConfig);
  }
  const configCancel = byId("config-cancel");
  if (configCancel) {
    configCancel.addEventListener("click", closeConfig);
  }
  const configModal = byId("config-modal");
  if (configModal) {
    configModal.addEventListener("click", (event) => {
      if (event.target === configModal) {
        closeConfig();
      }
    });
  }

  /* -------------------------------------------------------------- composer */

  const composer = byId("composer");
  if (composer) {
    let current = null;
    let source = null;

    function api(workId, movementId, suffix) {
      return `/api/works/${workId}/movements/${movementId}/${suffix}`;
    }

    function exportUrl(workId, movementId, fmt) {
      return `/works/${workId}/movements/${movementId}/export/${fmt}`;
    }

    function setProgressVisible(visible) {
      const panel = byId("progress-panel");
      if (panel) {
        panel.classList.toggle("hidden", !visible);
      }
    }

    function revealPanel(id) {
      const panel = byId(id);
      if (panel) {
        panel.classList.remove("hidden");
      }
    }

    function clearProgress() {
      const progress = byId("progress-list");
      if (progress) {
        progress.innerHTML = "";
      }
      const violations = byId("violation-list");
      if (violations) {
        violations.innerHTML = "";
      }
      const summary = byId("agent-summary");
      if (summary) {
        summary.textContent = "";
      }
      const checker = byId("checker-inline");
      if (checker) {
        checker.classList.add("hidden");
        checker.innerHTML = "";
      }
      const checkerPanel = byId("checker-panel");
      if (checkerPanel) {
        checkerPanel.classList.add("hidden");
      }
      const plan = byId("plan-inline");
      if (plan) {
        plan.classList.add("hidden");
        plan.innerHTML = "";
      }
      const planPanel = byId("plan-panel");
      if (planPanel) {
        planPanel.classList.add("hidden");
      }
    }

    function addProgress(text, className) {
      const item = document.createElement("li");
      item.className = "progress-item" + (className ? ` ${className}` : "");
      item.textContent = text;
      const progress = byId("progress-list");
      if (progress) {
        progress.appendChild(item);
      }
      return item;
    }

    function addDetail(label, text, className) {
      const value = String(text || "").trim();
      if (!value) {
        return;
      }
      const item = addProgress(label, className);
      const body = document.createElement("div");
      body.className = "detail-text";
      body.textContent = value;
      item.appendChild(body);
    }

    function toolLabel(name) {
      const labels = {
        submit_theme: "提交主题",
        add_part: "新增声部",
        edit: "编辑小节",
      };
      if (labels[name]) {
        return labels[name];
      }
      if (name && name.indexOf("technique_") === 0) {
        return `技法 · ${name.slice("technique_".length)}`;
      }
      return name || "工具";
    }

    function renderEvent(event) {
      if (event.kind === "start") {
        addProgress("智能体开始工作…", "running");
      } else if (event.kind === "thinking") {
        addProgress(`第 ${event.step} 轮思考…`, "running");
      } else if (event.kind === "tool_call") {
        addProgress(`调用 ${toolLabel(event.tool)}`, "running");
      } else if (event.kind === "tool_result") {
        if (event.ok === false && Array.isArray(event.violations) && event.violations.length) {
          event.violations.forEach((item) => {
            const who = item.voice_b
              ? `${item.voice_a} 与 ${item.voice_b}`
              : item.voice_a || "全体声部";
            addProgress(
              `${toolLabel(event.tool)} · 小节 ${item.measure} · ${who}：${item.message_zh}`,
              "bad"
            );
          });
        } else if (event.ok === false) {
          addProgress(`${toolLabel(event.tool)} 失败：${(event.message || "").slice(0, 80)}`, "bad");
        } else {
          addProgress(`${toolLabel(event.tool)} 完成：${(event.message || "").slice(0, 80)}`, "ok");
        }
        refreshScore();
      } else if (event.kind === "plan_start") {
        addProgress("创作AI 规划全部乐器与情感走向…", "running");
        renderPlanPending();
      } else if (event.kind === "plan") {
        addProgress("创作规划完成。", "ok");
        renderPlan(event.text, event.tree);
      } else if (event.kind === "movement_start") {
        const label = event.name ? `（${event.name}）` : "";
        addProgress(`开始创作乐章 ${event.movement}${label}…`, "running");
      } else if (event.kind === "instruction") {
        const prefix = event.movement ? `乐章 ${event.movement} · ` : "";
        addDetail(`${prefix}发给创作AI 的提示：`, event.text, "prompt");
      } else if (event.kind === "feedback") {
        const who = event.layer === "reviewer" ? "检查AI" : "符号层";
        const prefix = event.movement ? `乐章 ${event.movement} · ` : "";
        addDetail(
          `${prefix}${who}反馈：`,
          event.text,
          event.layer === "reviewer" ? "running" : "bad"
        );
      } else if (event.kind === "movement_done") {
        addProgress(`乐章 ${event.movement} 创作完成。`, "ok");
        refreshScore();
      } else if (event.kind === "movement_fix") {
        addProgress(`乐章 ${event.movement} 未通过检查，正在返工…`, "bad");
      } else if (event.kind === "review_start") {
        const label = event.movement ? `乐章 ${event.movement}` : "作品";
        addProgress(`检查AI开始评审${label}…`, "running");
        renderCheckerPending();
      } else if (event.kind === "review") {
        const verdict = event.passed ? "通过" : "打回";
        const detail = (event.suggestions || "").slice(0, 160);
        const label = event.movement ? `乐章 ${event.movement}：` : "";
        addProgress(
          `${label}检查AI ${verdict}${detail ? "：" + detail : ""}`,
          event.passed ? "ok" : "bad"
        );
        renderChecker(event.passed, event.suggestions);
      } else if (event.kind === "compress") {
        addProgress("会话接近上下文上限，已自动压缩后继续。", "running");
      } else if (event.kind === "assistant") {
        const text = String(event.text || "").trim();
        if (text) {
          const item = addProgress("创作AI 反馈：", "assistant");
          const body = document.createElement("div");
          body.className = "assistant-text";
          body.textContent = text;
          item.appendChild(body);
        }
        if (event.ok === false && Array.isArray(event.violations) && event.violations.length) {
          addProgress("符号层未通过。", "bad");
          event.violations.forEach((item) => {
            const who = item.voice_b
              ? `${item.voice_a} 与 ${item.voice_b}`
              : item.voice_a || "全体声部";
            addProgress(`小节 ${item.measure} · ${who}：${item.message_zh}`, "bad");
          });
        }
        refreshScore();
      } else if (event.kind === "error") {
        addProgress(`出错：${event.message}`, "bad");
      }
    }

    function renderPlanPending() {
      const box = byId("plan-inline");
      if (!box) {
        return;
      }
      revealPanel("plan-panel");
      box.classList.remove("hidden");
      box.innerHTML = "";
      const head = document.createElement("div");
      head.className = "plan-pending";
      head.textContent = "创作AI 规划中…";
      box.appendChild(head);
    }

    function renderPlanPrompts(box, tree) {
      const list = document.createElement("div");
      list.className = "plan-prompts";
      tree.forEach((movement) => {
        const head = document.createElement("div");
        head.className = "plan-movement";
        head.textContent = `乐章 ${movement.index} · ${movement.name}`;
        list.appendChild(head);
        const item = document.createElement("div");
        item.className = "plan-section";
        const prompt = document.createElement("div");
        prompt.className = "plan-section-prompt";
        prompt.textContent = movement.prompt || "(未填写)";
        item.appendChild(prompt);
        list.appendChild(item);
      });
      box.appendChild(list);
    }

    function renderPlan(text, tree) {
      const box = byId("plan-inline");
      if (!box || (!text && !(tree && tree.length))) {
        return;
      }
      revealPanel("plan-panel");
      box.classList.remove("hidden");
      box.innerHTML = "";
      const head = document.createElement("div");
      head.className = "plan-title";
      const label = document.createElement("span");
      label.textContent = "创作规划";
      head.appendChild(label);
      const copy = document.createElement("button");
      copy.type = "button";
      copy.className = "btn tiny";
      copy.textContent = "复制规划";
      copy.addEventListener("click", () => {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text || "").then(
            () => showToast("已复制创作规划。", "ok"),
            () => showToast("复制失败，请手动选择文字复制。", "bad")
          );
        } else {
          showToast("复制失败，请手动选择文字复制。", "bad");
        }
      });
      head.appendChild(copy);
      box.appendChild(head);
      if (tree && tree.length) {
        renderPlanPrompts(box, tree);
      } else {
        const body = document.createElement("div");
        body.className = "plan-detail";
        body.textContent = text;
        box.appendChild(body);
      }
    }

    function renderCheckerPending() {
      const box = byId("checker-inline");
      if (!box) {
        return;
      }
      revealPanel("checker-panel");
      box.classList.remove("hidden");
      box.innerHTML = "";
      const head = document.createElement("div");
      head.className = "checker-pending";
      head.textContent = "检查AI 评审中…";
      box.appendChild(head);
    }

    function renderChecker(passed, suggestions) {
      const box = byId("checker-inline");
      if (!box) {
        return;
      }
      revealPanel("checker-panel");
      box.classList.remove("hidden");
      box.innerHTML = "";
      const head = document.createElement("div");
      head.className = "checker-head";
      const verdict = document.createElement("span");
      verdict.className = passed ? "checker-pass" : "checker-fail";
      verdict.textContent = passed ? "✔ 检查AI 通过" : "✘ 检查AI 打回";
      head.appendChild(verdict);
      if (suggestions) {
        const copy = document.createElement("button");
        copy.type = "button";
        copy.className = "btn tiny";
        copy.textContent = "复制评审";
        copy.addEventListener("click", () => {
          if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(suggestions).then(
              () => showToast("已复制评审内容。", "ok"),
              () => showToast("复制失败，请手动选择文字复制。", "bad")
            );
          } else {
            showToast("复制失败，请手动选择文字复制。", "bad");
          }
        });
        head.appendChild(copy);
      }
      box.appendChild(head);
      if (!passed && suggestions) {
        const detail = document.createElement("div");
        detail.className = "checker-detail";
        detail.textContent = suggestions;
        box.appendChild(detail);
      }
    }

    function renderViolations(items) {
      const list = byId("violation-list");
      if (!list) {
        return;
      }
      list.innerHTML = "";
      if (!items || !items.length) {
        return;
      }
      items.forEach((item) => {
        const li = document.createElement("li");
        li.className = "bad";
        const who = item.voice_b
          ? `${item.voice_a} 与 ${item.voice_b}`
          : item.voice_a || "全体声部";
        li.textContent = `小节 ${item.measure} · ${who}：${item.message_zh}`;
        list.appendChild(li);
      });
    }

    function showResult(workId, movementId, title, status, violations) {
      current = { workId, movementId };
      const panel = byId("result-panel");
      if (panel) {
        panel.classList.remove("hidden");
      }
      const titleNode = byId("result-title");
      if (title && titleNode) {
        titleNode.textContent = title;
      }
      const statusNode = byId("result-status");
      if (status && statusNode) {
        statusNode.textContent = status;
      }
      setHref("export-musicxml", exportUrl(workId, movementId, "musicxml"));
      setHref("export-m4a", exportUrl(workId, movementId, "m4a"));
      setHref("export-mp3", exportUrl(workId, movementId, "mp3"));
      renderViolations(violations || []);
      playAudio(workId, movementId);
    }

    function playAudio(workId, movementId) {
      const audio = byId("audition");
      const hint = byId("audio-hint");
      if (!audio) {
        return;
      }
      audio.src = exportUrl(workId, movementId, "m4a");
      audio.load();
      audio
        .play()
        .then(() => {
          if (hint) {
            hint.textContent = "";
          }
        })
        .catch(() => {
          if (hint) {
            hint.textContent =
              "无法自动播放（可能缺少音频合成组件），请点击播放器手动播放或导出 M4A/MP3。";
          }
        });
    }

    function loadScore(workId, movementId) {
      const container = byId("score-view");
      if (!container) {
        return;
      }
      const panel = byId("score-panel");
      if (panel) {
        panel.classList.remove("hidden");
      }
      fetch(api(workId, movementId, "score"))
        .then((response) => response.text())
        .then((xml) => renderScoreInto(container, xml))
        .catch(() => {
          container.textContent = "乐谱加载失败。";
        });
    }

    let lastScoreRefresh = 0;

    function refreshScore() {
      if (!current) {
        return;
      }
      const now = Date.now();
      if (now - lastScoreRefresh < 1200) {
        return;
      }
      lastScoreRefresh = now;
      const panel = byId("score-panel");
      if (panel) {
        panel.classList.remove("hidden");
      }
      loadScore(current.workId, current.movementId);
    }

    function streamJob(jobId) {
      if (source) {
        source.close();
      }
      source = new EventSource(`/api/jobs/${jobId}/events`);
      source.onmessage = (message) => {
        try {
          renderEvent(JSON.parse(message.data));
        } catch (error) {
          /* ignore malformed event */
        }
      };
      source.addEventListener("done", (message) => {
        source.close();
        source = null;
        const result = JSON.parse(message.data);
        const button = byId("generate-btn");
        if (button) {
          button.disabled = false;
          button.textContent = "生成";
        }
        if (result.status === "error") {
          addProgress(`任务失败：${result.message || "未知错误"}`, "bad");
          return;
        }
        addProgress(
          result.ok ? "符号层校验通过。" : "符号层仍有违规，可提意见继续修改。",
          result.ok ? "ok" : "bad"
        );
        const summary = byId("agent-summary");
        if (summary) {
          summary.textContent = result.final_text || "";
        }
        if (result.plan || (result.plan_tree && result.plan_tree.length)) {
          renderPlan(result.plan, result.plan_tree);
        }
        if (result.review_passed !== null && result.review_passed !== undefined) {
          renderChecker(result.review_passed, result.review_suggestions);
        }
        showResult(
          result.work_id,
          result.movement_id,
          null,
          result.ok ? "已校验" : "草稿",
          result.violations
        );
        loadScore(result.work_id, result.movement_id);
      });
      source.onerror = () => {
        if (source) {
          source.close();
          source = null;
        }
      };
    }

    function setButtonBusy(button, busy) {
      if (!button) {
        return;
      }
      button.disabled = busy;
      button.textContent = busy ? "生成中…" : "生成";
    }

    function startJob(url, body, button) {
      setButtonBusy(button, true);
      setProgressVisible(true);
      clearProgress();
      const panel = byId("result-panel");
      if (panel) {
        panel.classList.add("hidden");
      }
      const scorePanel = byId("score-panel");
      if (scorePanel) {
        scorePanel.classList.remove("hidden");
      }
      const score = byId("score-view");
      if (score) {
        score.innerHTML = "";
      }
      fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
        .then((response) => response.json())
        .then((data) => {
          if (!data.ok) {
            addProgress(data.error || "无法开始生成。", "bad");
            setButtonBusy(button, false);
            return;
          }
          current = { workId: data.work_id, movementId: data.movement_id };
          streamJob(data.job_id);
        })
        .catch(() => {
          addProgress("请求失败。", "bad");
          setButtonBusy(button, false);
        });
    }

    on("generate-btn", "click", () => {
      const promptNode = byId("prompt");
      const prompt = promptNode ? promptNode.value.trim() : "";
      if (!prompt) {
        window.alert("请先输入创作提示词。");
        return;
      }
      const genreNode = byId("genre");
      startJob(
        "/api/generate",
        { prompt, genre: genreNode ? genreNode.value : "plain" },
        byId("generate-btn")
      );
    });

    on("score-btn", "click", () => {
      if (current) {
        loadScore(current.workId, current.movementId);
      }
    });

    on("copy-score-btn", "click", () => {
      if (!current) {
        showToast("尚无乐谱可复制。", "bad");
        return;
      }
      copyScore(api(current.workId, current.movementId, "score"));
    });

    on("export-png", "click", () => {
      if (current) {
        downloadScorePng(byId("score-view"), `${current.workId}-${current.movementId}.png`);
      }
    });

    on("finalize-btn", "click", () => {
      if (!current) {
        return;
      }
      fetch(api(current.workId, current.movementId, "finalize"), { method: "POST" })
        .then((response) => response.json())
        .then((data) => {
          const statusNode = byId("result-status");
          if (data.ok && statusNode) {
            statusNode.textContent = "已定稿";
          }
          window.alert(data.message || "已处理。");
        });
    });

    on("revise-btn", "click", () => {
      const area = byId("feedback-area");
      if (area) {
        area.classList.toggle("hidden");
      }
    });

    on("feedback-btn", "click", () => {
      if (!current) {
        return;
      }
      const feedbackNode = byId("feedback-text");
      const feedback = feedbackNode ? feedbackNode.value.trim() : "";
      if (!feedback) {
        window.alert("请先输入修改意见。");
        return;
      }
      startJob(
        api(current.workId, current.movementId, "run"),
        { prompt: "根据品鉴意见继续完善这首作品。", feedback },
        byId("generate-btn")
      );
    });

    document.querySelectorAll("#history-list .history-link").forEach((link) => {
      link.addEventListener("click", (event) => {
        event.preventDefault();
        const card = link.closest(".work-card");
        if (!card) {
          return;
        }
        const workId = card.dataset.work;
        const movementId = card.dataset.movement;
        const titleNode = card.querySelector("strong");
        const statusNode = card.querySelector(".chip.state");
        showResult(
          workId,
          movementId,
          titleNode ? titleNode.textContent : "",
          statusNode ? statusNode.textContent : "",
          []
        );
        loadScore(workId, movementId);
        const panel = byId("result-panel");
        if (panel) {
          window.scrollTo({ top: panel.offsetTop, behavior: "smooth" });
        }
      });
    });
  }

  /* ------------------------------------------------------------- work page */

  const workspace = document.querySelector(".workspace");
  if (workspace) {
    const workId = workspace.dataset.work;
    const movementId = workspace.dataset.movement;
    const base = `/api/works/${workId}/movements/${movementId}`;
    const scoreView = byId("score-view");

    function openDrawer(drawer) {
      drawer.classList.add("open");
    }

    function loadWorkScore() {
      if (!scoreView) {
        return;
      }
      fetch(`${base}/score`)
        .then((response) => response.text())
        .then((xml) => renderScoreInto(scoreView, xml))
        .catch(() => {
          scoreView.textContent = "乐谱加载失败。";
        });
    }

    loadWorkScore();

    const workScoreBtn = byId("work-score-btn");
    if (workScoreBtn) {
      workScoreBtn.addEventListener("click", loadWorkScore);
    }

    const workPngBtn = byId("work-png-btn");
    if (workPngBtn) {
      workPngBtn.addEventListener("click", () => {
        downloadScorePng(scoreView, `${workId}-${movementId}.png`);
      });
    }

    const workCopyBtn = byId("work-copy-btn");
    if (workCopyBtn) {
      workCopyBtn.addEventListener("click", () => {
        copyScore(`${base}/score`);
      });
    }

    function renderWorkViolations(items) {
      const list = byId("violation-list");
      if (!list) {
        return;
      }
      list.innerHTML = "";
      if (!items.length) {
        const li = document.createElement("li");
        li.className = "ok";
        li.textContent = "检查通过，未发现违规。";
        list.appendChild(li);
        return;
      }
      items.forEach((item) => {
        const li = document.createElement("li");
        const who = item.voice_b
          ? `${item.voice_a} 与 ${item.voice_b}`
          : item.voice_a || "全体声部";
        li.textContent = `小节 ${item.measure} · ${who}：${item.message_zh}`;
        list.appendChild(li);
      });
    }

    const checkBtn = byId("check-btn");
    if (checkBtn) {
      checkBtn.addEventListener("click", () => {
        fetch(`${base}/check`, { method: "POST" })
          .then((response) => response.json())
          .then((data) => {
            renderWorkViolations(data.violations || []);
            openDrawer(byId("check-drawer"));
            const audio = byId("audition");
            if (audio) {
              audio.load();
            }
          });
      });
    }

    const finalizeBtn = byId("finalize-btn");
    if (finalizeBtn) {
      finalizeBtn.addEventListener("click", () => {
        fetch(`${base}/finalize`, { method: "POST" })
          .then((response) => response.json())
          .then((data) => {
            const chip = byId("status-chip");
            if (data.ok && chip) {
              chip.textContent = "已定稿";
            }
            window.alert(data.message || "已处理。");
          });
      });
    }

    document.querySelectorAll(".rollback").forEach((button) => {
      button.addEventListener("click", () => {
        fetch(`${base}/rollback`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ seq: Number(button.dataset.seq) }),
        })
          .then((response) => response.json())
          .then((data) => {
            if (data.ok) {
              window.location.reload();
            } else {
              window.alert(data.message || "回退失败。");
            }
          });
      });
    });

    const auditBtn = byId("audit-btn");
    if (auditBtn) {
      auditBtn.addEventListener("click", () => {
        const note = byId("audit-note").value;
        fetch(`${base}/audit`, { method: "POST", body: new URLSearchParams({ note }) })
          .then((response) => response.json())
          .then((data) => {
            byId("audit-status").textContent = data.ok
              ? "已记录，可运行智能体继续修改。"
              : "提交失败。";
          });
      });
    }
  }
})();
