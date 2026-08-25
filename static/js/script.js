// DOM Elements
const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const sendButton = document.getElementById("sendButton");

/**
 * Add a message to the chat
 * @param {string} content - Message content
 * @param {boolean} isUser - Whether the message is from user
 */
function addMessage(content, isUser = false) {
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${isUser ? "user" : "bot"}`;

  const avatarEl = document.createElement("div");
  avatarEl.className = "message-avatar";
  avatarEl.textContent = isUser ? "👤" : "🤖";

  const contentEl = document.createElement("div");
  contentEl.className = "message-content";

  // Render markdown for bot messages; use textContent for user input to prevent XSS.
  if (!isUser) {
    if (
      window.MarkdownRenderer &&
      typeof window.MarkdownRenderer.renderToElement === "function"
    ) {
      contentEl.appendChild(window.MarkdownRenderer.renderToElement(content));
    } else {
      // Fallback: plain text (should be rare if markdown.js is loaded)
      contentEl.textContent = content;
      contentEl.style.whiteSpace = "pre-wrap";
    }
  } else {
    contentEl.textContent = content;
  }

  messageDiv.appendChild(avatarEl);
  messageDiv.appendChild(contentEl);
  chatMessages.appendChild(messageDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Show typing indicator
 */
function showTypingIndicator() {
  const typingDiv = document.createElement("div");
  typingDiv.className = "message bot typing-indicator";
  typingDiv.id = "typing";
  typingDiv.innerHTML = `
    <div class="message-avatar">🤖</div>
    <div class="message-content">...</div>
  `;
  chatMessages.appendChild(typingDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Remove typing indicator
 */
function removeTypingIndicator() {
  const typingDiv = document.getElementById("typing");
  if (typingDiv) {
    typingDiv.remove();
  }
}

let currentController = null;

/**
 * Get bot response from API
 * @param {string} userMessage - User's message
 */
async function getBotResponse(userMessage) {
  if (currentController) {
    currentController.abort();
  }
  currentController = new AbortController();

  try {
    showTypingIndicator();

    const response = await fetch("/response", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: userMessage }),
      signal: currentController.signal,
    });

    removeTypingIndicator();

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Failed to get response");
    }

    const data = await response.json();
    addMessage(data.response, false);
  } catch (error) {
    removeTypingIndicator();
    if (error.name === "AbortError") return;
    console.error("Error:", error);
    addMessage(
      `抱歉，發生錯誤：${error.message}。請確認後端伺服器是否正在運行。`,
      false,
    );
  } finally {
    currentController = null;
  }
}

/**
 * Handle form submission
 * @param {Event} e - Submit event
 */
function handleSubmit(e) {
  e.preventDefault();

  const message = chatInput.value.trim();
  if (message) {
    addMessage(message, true);
    chatInput.value = "";
    sendButton.disabled = true;

    getBotResponse(message).finally(() => {
      sendButton.disabled = false;
      chatInput.focus();
    });
  }
}

// Event Listeners
chatForm.addEventListener("submit", handleSubmit);

// Auto-focus input on load
chatInput.focus();
