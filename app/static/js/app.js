/*
SPDX-FileCopyrightText: 2026 xhdlphzr
SPDX-License-Identifier: MIT
*/

(function () {
  "use strict";

  function t(text, vars) {
    let out = (window.I18N && window.I18N[text]) || text;
    if (vars) {
      Object.keys(vars).forEach((key) => {
        out = out.replace("{" + key + "}", String(vars[key]));
      });
    }
    return out;
  }

  function violationText(item) {
    const key = "violation." + item.kind;
    const template = window.I18N && window.I18N[key];
    if (!template) {
      return item.message || "";
    }
    return t(key, {
      measure: item.measure,
      voice_a: item.voice_a || "",
      voice_b: item.voice_b || "",
    });
  }

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
          showToast(t("js.no_score_copy"), "bad");
          return;
        }
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(xml).then(
            () => showToast(t("js.copied_musicxml"), "ok"),
            () => showToast(t("js.copy_failed"), "bad")
          );
        } else {
          showToast(t("js.copy_failed"), "bad");
        }
      })
      .catch(() => showToast(t("js.read_score_failed"), "bad"));
  }

  document.addEventListener("click", (event) => {
    const target = event.target;
    const link = target && target.closest ? target.closest("a.export-download") : null;
    if (!link) {
      return;
    }
    event.preventDefault();
    const base = link.href.split("?")[0];
    showToast(t("js.exporting"), "ok");
    fetch(`${base}?save=1`)
      .then((response) => response.json())
      .then((data) => {
        if (data.ok) {
          showToast(t("js.exported", { path: data.path || "" }), "ok");
        } else {
          showToast(data.error || t("js.export_failed"), "bad");
        }
      })
      .catch(() => showToast(t("js.export_failed"), "bad"));
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
      container.textContent = t("js.no_score");
      return Promise.resolve();
    }
    if (text.indexOf("<?xml") !== 0) {
      container.textContent = t("js.invalid_score");
      if (window.console) {
        window.console.error("Invalid MusicXML payload:", text.slice(0, 160));
      }
      return Promise.resolve();
    }
    if (typeof opensheetmusicdisplay === "undefined") {
      container.textContent = t("js.renderer_missing");
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
          container.textContent = t("js.render_failed");
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
      window.alert(t("js.show_score_first"));
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
      showToast(t("js.exported", { path: filename }), "ok");
    };
    image.onerror = () => showToast(t("js.png_failed"), "bad");
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

  function loadCustomKits() {
    const list = byId("custom-kit-list");
    if (!list) {
      return;
    }
    fetch("/api/kits")
      .then((response) => response.json())
      .then((data) => {
        list.innerHTML = "";
        const custom = (data.kits || []).filter((kit) => !kit.builtin);
        if (!custom.length) {
          const empty = document.createElement("li");
          empty.className = "muted small";
          empty.textContent = t("style.no_custom");
          list.appendChild(empty);
          return;
        }
        custom.forEach((kit) => {
          const item = document.createElement("li");
          const label = document.createElement("span");
          label.textContent = kit.name;
          const rename = document.createElement("button");
          rename.type = "button";
          rename.className = "btn tiny";
          rename.textContent = t("style.rename");
          rename.addEventListener("click", () => renameKit(kit));
          const remove = document.createElement("button");
          remove.type = "button";
          remove.className = "btn tiny";
          remove.textContent = t("style.delete");
          remove.addEventListener("click", () => deleteKit(kit));
          item.append(label, rename, remove);
          list.appendChild(item);
        });
      })
      .catch(() => {});
  }

  function renameKit(kit) {
    const name = window.prompt(t("style.rename"), kit.name);
    if (!name || !name.trim()) {
      return;
    }
    fetch(`/api/kits/${kit.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name.trim() }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.ok) {
          window.location.reload();
        }
      });
  }

  function deleteKit(kit) {
    if (!window.confirm(t("style.delete_confirm"))) {
      return;
    }
    fetch(`/api/kits/${kit.id}`, { method: "DELETE" })
      .then((response) => response.json())
      .then((data) => {
        if (data.ok) {
          window.location.reload();
        }
      });
  }

  function openConfig() {
    loadCustomKits();
    fetch("/api/config")
      .then((response) => response.json())
      .then((config) => {
        setFieldValue("cfg-base-url", config.base_url);
        setFieldValue("cfg-api-key", config.api_key);
        setFieldValue("cfg-model", config.model);
        setFieldValue("cfg-context-window", config.context_window);
        setFieldValue("cfg-language", config.language || "en");
        const modal = byId("config-modal");
        if (modal) {
          modal.classList.remove("hidden");
        }
      })
      .catch(() => window.alert(t("js.config_read_failed")));
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
      language: fieldValue("cfg-language") || "en",
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
          window.location.reload();
        } else {
          window.alert(data.error || t("js.save_failed"));
        }
      })
      .catch(() => window.alert(t("js.save_failed")));
  }

  function setLanguage(language) {
    fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ language }),
    })
      .then(() => window.location.reload())
      .catch(() => window.location.reload());
  }

  const configBtn = byId("config-btn");
  if (configBtn) {
    configBtn.addEventListener("click", openConfig);
  }
  const langSelect = byId("lang-select");
  if (langSelect) {
    langSelect.addEventListener("change", () => setLanguage(langSelect.value));
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

  /* --------------------------------------------------- llm connectivity */

  const LLM_BUTTONS = ["generate-btn", "feedback-btn"];
  let llmConnected = false;

  function applyLlmButtons() {
    LLM_BUTTONS.forEach((id) => {
      const button = byId(id);
      if (button) {
        button.disabled = !llmConnected;
      }
    });
  }

  function applyLlmStatus(status) {
    const dot = byId("llm-status-dot");
    const text = byId("llm-status-text");
    if (dot) {
      dot.className = "llm-dot";
      if (status === "ok") {
        dot.classList.add("llm-dot-ok");
      } else if (status === "error") {
        dot.classList.add("llm-dot-error");
      } else {
        dot.classList.add("llm-dot-checking");
      }
    }
    if (text) {
      text.textContent = status === "error" ? t("llm.failed") : "";
    }
    llmConnected = status === "ok";
    applyLlmButtons();
  }

  function probeLlm(url, options) {
    const dot = byId("llm-status-dot");
    if (dot) {
      dot.className = "llm-dot llm-dot-checking";
    }
    fetch(url, options)
      .then((response) => response.json())
      .then((data) => applyLlmStatus(data.status))
      .catch(() => applyLlmButtons());
  }

  if (byId("llm-status-dot")) {
    applyLlmButtons();
    on("llm-ping-btn", "click", () => probeLlm("/api/llm-ping", { method: "POST" }));
    probeLlm("/api/llm-status");
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
        submit_theme: t("tool.submit"),
        add_part: t("tool.add_part"),
        edit: t("tool.edit"),
      };
      if (labels[name]) {
        return labels[name];
      }
      if (name && name.indexOf("technique_") === 0) {
        return t("tool.technique_name", { name: name.slice("technique_".length) });
      }
      return name || t("tool.unknown");
    }

    function renderEvent(event) {
      if (event.kind === "start") {
        addProgress(t("progress.start"), "running");
      } else if (event.kind === "thinking") {
        addProgress(t("progress.thinking", { step: event.step }), "running");
      } else if (event.kind === "tool_call") {
        addProgress(t("progress.tool_call", { tool: toolLabel(event.tool) }), "running");
      } else if (event.kind === "tool_result") {
        if (event.ok === false && Array.isArray(event.violations) && event.violations.length) {
          event.violations.forEach((item) => {
            const who = item.voice_b
              ? t("progress.and", { a: item.voice_a, b: item.voice_b })
              : item.voice_a || t("progress.all_voices");
            addProgress(
              t("progress.violation_tool", {
                tool: toolLabel(event.tool),
                measure: item.measure,
                who,
                message: violationText(item),
              }),
              "bad"
            );
          });
        } else if (event.ok === false) {
          addProgress(
            t("progress.tool_failed", {
              tool: toolLabel(event.tool),
              message: (event.message || "").slice(0, 80),
            }),
            "bad"
          );
        } else {
          addProgress(
            t("progress.tool_done", {
              tool: toolLabel(event.tool),
              message: (event.message || "").slice(0, 80),
            }),
            "ok"
          );
        }
        refreshScore();
      } else if (event.kind === "plan_start") {
        addProgress(t("progress.plan_start"), "running");
        renderPlanPending();
      } else if (event.kind === "plan") {
        addProgress(t("progress.plan_done"), "ok");
        renderPlan(event.text, event.tree);
      } else if (event.kind === "movement_start") {
        const label = event.name ? `（${event.name}）` : "";
        addProgress(t("progress.movement_start", { movement: event.movement, label }), "running");
      } else if (event.kind === "instruction") {
        const prefix = event.movement ? t("progress.movement_prefix", { movement: event.movement }) : "";
        addDetail(`${prefix}${t("progress.prompt_sent")}`, event.text, "prompt");
      } else if (event.kind === "feedback") {
        const who = event.layer === "reviewer" ? t("layer.reviewer") : t("layer.symbolic");
        const prefix = event.movement ? t("progress.movement_prefix", { movement: event.movement }) : "";
        addDetail(
          `${prefix}${t("feedback.layer", { who })}`,
          event.text,
          event.layer === "reviewer" ? "running" : "bad"
        );
      } else if (event.kind === "movement_done") {
        addProgress(t("progress.movement_done", { movement: event.movement }), "ok");
        refreshScore();
      } else if (event.kind === "movement_fix") {
        addProgress(t("progress.movement_fix", { movement: event.movement }), "bad");
      } else if (event.kind === "review_start") {
        const label = event.movement
          ? t("progress.movement", { movement: event.movement })
          : t("result.work");
        addProgress(t("progress.review_start", { label }), "running");
        renderCheckerPending();
      } else if (event.kind === "review") {
        const verdict = event.passed ? t("verdict.passed") : t("verdict.rejected");
        const detail = (event.suggestions || "").slice(0, 160);
        const label = event.movement
          ? t("progress.movement_colon", { movement: event.movement })
          : "";
        addProgress(
          t("progress.review", {
            label,
            verdict,
            detail: detail ? "：" + detail : "",
          }),
          event.passed ? "ok" : "bad"
        );
        renderChecker(event.passed, event.suggestions);
      } else if (event.kind === "compress") {
        addProgress(t("progress.compressed"), "running");
      } else if (event.kind === "assistant") {
        const text = String(event.text || "").trim();
        if (text) {
          const item = addProgress(t("progress.assistant"), "assistant");
          const body = document.createElement("div");
          body.className = "assistant-text";
          body.textContent = text;
          item.appendChild(body);
        }
        if (event.ok === false && Array.isArray(event.violations) && event.violations.length) {
          addProgress(t("progress.symbolic_failed"), "bad");
          event.violations.forEach((item) => {
            const who = item.voice_b
              ? t("progress.and", { a: item.voice_a, b: item.voice_b })
              : item.voice_a || t("progress.all_voices");
            addProgress(
              t("progress.violation", {
                measure: item.measure,
                who,
                message: violationText(item),
              }),
              "bad"
            );
          });
        }
        refreshScore();
      } else if (event.kind === "error") {
        addProgress(t("progress.error", { message: event.message }), "bad");
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
      head.textContent = t("plan.pending");
      box.appendChild(head);
    }

    function renderPlanPrompts(box, tree) {
      const list = document.createElement("div");
      list.className = "plan-prompts";
      tree.forEach((movement) => {
        const head = document.createElement("div");
        head.className = "plan-movement";
        head.textContent = t("plan.movement", {
          index: movement.index,
          name: movement.name,
        });
        list.appendChild(head);
        const item = document.createElement("div");
        item.className = "plan-section";
        const prompt = document.createElement("div");
        prompt.className = "plan-section-prompt";
        prompt.textContent = movement.prompt || t("plan.empty_cell");
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
      label.textContent = t("plan.title");
      head.appendChild(label);
      const copy = document.createElement("button");
      copy.type = "button";
      copy.className = "btn tiny";
      copy.textContent = t("plan.copy");
      copy.addEventListener("click", () => {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text || "").then(
            () => showToast(t("plan.copied"), "ok"),
            () => showToast(t("js.copy_failed"), "bad")
          );
        } else {
          showToast(t("js.copy_failed"), "bad");
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
      head.textContent = t("checker.pending");
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
      verdict.textContent = passed ? t("checker.passed") : t("checker.rejected");
      head.appendChild(verdict);
      if (suggestions) {
        const copy = document.createElement("button");
        copy.type = "button";
        copy.className = "btn tiny";
        copy.textContent = t("checker.copy");
        copy.addEventListener("click", () => {
          if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(suggestions).then(
              () => showToast(t("checker.copied"), "ok"),
              () => showToast(t("js.copy_failed"), "bad")
            );
          } else {
            showToast(t("js.copy_failed"), "bad");
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
          ? t("progress.and", { a: item.voice_a, b: item.voice_b })
          : item.voice_a || t("progress.all_voices");
        li.textContent = t("progress.violation", {
          measure: item.measure,
          who,
          message: violationText(item),
        });
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
              t("js.autoplay_failed");
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
          container.textContent = t("js.score_load_failed");
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
          button.textContent = t("composer.generate");
        }
        if (result.status === "error") {
          addProgress(
            t("js.job_failed", { message: result.message || t("common.unknown_error") }),
            "bad"
          );
          return;
        }
        addProgress(
          result.ok ? t("progress.check_passed") : t("progress.check_violations"),
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
          result.ok ? t("status.checked") : t("status.draft"),
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
      button.textContent = busy ? t("js.generating") : t("composer.generate");
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
            addProgress(data.error || t("js.cannot_start"), "bad");
            setButtonBusy(button, false);
            return;
          }
          current = { workId: data.work_id, movementId: data.movement_id };
          streamJob(data.job_id);
        })
        .catch(() => {
          addProgress(t("js.request_failed"), "bad");
          setButtonBusy(button, false);
        });
    }

    const styleSelect = byId("style");
    const styleCustom = byId("style-custom");
    if (styleSelect && styleCustom) {
      styleSelect.addEventListener("change", () => {
        styleCustom.classList.toggle("hidden", styleSelect.value !== "__custom__");
      });
      document.querySelectorAll(".style-tab").forEach((tab) => {
        tab.addEventListener("click", () => {
          document.querySelectorAll(".style-tab").forEach((item) => {
            item.classList.remove("active");
          });
          tab.classList.add("active");
          const target = tab.dataset.tab;
          const rulesPane = byId("style-rules-pane");
          const techniquesPane = byId("style-techniques-pane");
          if (rulesPane) {
            rulesPane.classList.toggle("hidden", target !== "rules");
          }
          if (techniquesPane) {
            techniquesPane.classList.toggle("hidden", target !== "techniques");
          }
        });
      });
      on("style-confirm", "click", () => {
        const nameNode = byId("style-name");
        const name = nameNode ? nameNode.value.trim() : "";
        if (!name) {
          window.alert(t("style.need_name"));
          return;
        }
        const rules = Array.from(document.querySelectorAll(".style-rule:checked")).map(
          (node) => node.value
        );
        const techniques = Array.from(
          document.querySelectorAll(".style-technique:checked")
        ).map((node) => node.value);
        fetch("/api/kits", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, rules, techniques }),
        })
          .then((response) => response.json())
          .then((data) => {
            if (!data.ok) {
              window.alert(data.error || t("style.need_name"));
              return;
            }
            const option = document.createElement("option");
            option.value = data.kit.id;
            option.textContent = data.kit.name;
            styleSelect.insertBefore(option, styleSelect.lastElementChild);
            styleSelect.value = data.kit.id;
            styleCustom.classList.add("hidden");
            showToast(t("style.created"), "ok");
          })
          .catch(() => window.alert(t("style.need_name")));
      });
    }

    on("generate-btn", "click", () => {
      const promptNode = byId("prompt");
      const prompt = promptNode ? promptNode.value.trim() : "";
      if (!prompt) {
        window.alert(t("js.enter_prompt"));
        return;
      }
      const genreNode = byId("genre");
      const style = styleSelect ? styleSelect.value : "";
      if (style === "__custom__") {
        window.alert(t("style.confirm_first"));
        return;
      }
      startJob(
        "/api/generate",
        { prompt, genre: genreNode ? genreNode.value : "plain", style },
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
        showToast(t("js.no_score_copy"), "bad");
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
            statusNode.textContent = t("status.final");
          }
          window.alert(data.message || t("js.done"));
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
        window.alert(t("js.enter_feedback"));
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
          scoreView.textContent = t("js.score_load_failed");
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
        li.textContent = t("js.check_clean");
        list.appendChild(li);
        return;
      }
      items.forEach((item) => {
        const li = document.createElement("li");
        const who = item.voice_b
          ? t("progress.and", { a: item.voice_a, b: item.voice_b })
          : item.voice_a || t("progress.all_voices");
        li.textContent = t("progress.violation", {
          measure: item.measure,
          who,
          message: violationText(item),
        });
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
              chip.textContent = t("status.final");
            }
            window.alert(data.message || t("js.done"));
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
              window.alert(data.message || t("js.revert_failed"));
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
              ? t("js.recorded")
              : t("js.submit_failed");
          });
      });
    }
  }
})();
