/**
 * Markdown rendering utilities.
 *
 * Exposes a global `MarkdownRenderer` with:
 * - `renderToElement(content)` -> HTMLElement
 */

(function () {
  /**
   * Normalize model text into markdown-friendly structure.
   * This helps when headings/lists are returned in a single line.
   * @param {string} content - Raw model content
   * @returns {string}
   */
  function normalizeMarkdown(content) {
    if (typeof content !== "string") return "";

    return content
      .trim()
      .replace(/\s+(#{1,6}\s)/g, "\n\n$1")
      .replace(/\s+-\s+/g, "\n- ");
  }

  /**
   * Escape html entities for safe fallback rendering.
   * @param {string} text
   * @returns {string}
   */
  function escapeHtml(text) {
    return text
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  /**
   * Lightweight markdown renderer for environments where marked is unavailable.
   * Supports headings, bullet lists, bold, and paragraphs.
   * @param {string} content - Normalized markdown-like text
   * @returns {string}
   */
  function renderFallbackMarkdown(content) {
    const lines = content.split(/\r?\n/);
    const chunks = [];
    let inList = false;

    const closeListIfNeeded = () => {
      if (inList) {
        chunks.push("</ul>");
        inList = false;
      }
    };

    for (const rawLine of lines) {
      const line = rawLine.trim();

      if (!line) {
        closeListIfNeeded();
        continue;
      }

      const safe = escapeHtml(line).replace(
        /\*\*(.*?)\*\*/g,
        "<strong>$1</strong>",
      );
      const headingMatch = safe.match(/^(#{1,6})\s+(.+)$/);
      const listMatch = safe.match(/^-\s+(.+)$/);

      if (headingMatch) {
        closeListIfNeeded();
        const level = headingMatch[1].length;
        chunks.push(`<h${level}>${headingMatch[2]}</h${level}>`);
        continue;
      }

      if (listMatch) {
        if (!inList) {
          chunks.push("<ul>");
          inList = true;
        }
        chunks.push(`<li>${listMatch[1]}</li>`);
        continue;
      }

      closeListIfNeeded();
      chunks.push(`<p>${safe}</p>`);
    }

    closeListIfNeeded();
    return chunks.join("\n");
  }

  /**
   * Render markdown content into a DOM element.
   * Uses `marked` if available, otherwise uses a safe fallback renderer.
   * @param {string} content
   * @returns {HTMLElement}
   */
  function renderToElement(content) {
    const normalized = normalizeMarkdown(content);

    const el = document.createElement("div");
    el.className = "markdown-body";

    try {
      if (typeof marked !== "undefined") {
        el.innerHTML = marked.parse(normalized);
      } else {
        el.innerHTML = renderFallbackMarkdown(normalized);
      }
    } catch (error) {
      console.error("Markdown rendering error:", error);
      el.innerHTML = renderFallbackMarkdown(normalized);
    }

    return el;
  }

  window.MarkdownRenderer = {
    renderToElement,
  };
})();
