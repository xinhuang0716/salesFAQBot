// DOM Elements
const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const sendButton = document.getElementById("sendButton");

// API Configuration
const API_BASE_URL = "http://localhost:8010";

/**
 * Add a message to the chat
 * @param {string} content - Message content
 * @param {boolean} isUser - Whether the message is from user
 */

function addMessage(content, isUser = false) {
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${isUser ? "user" : "bot"}`;

  // Render markdown for bot messages, plain text for user messages
  let renderedContent;
  if (!isUser && typeof marked !== "undefined") {
    try {
      renderedContent = marked.parse(content);
      messageDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content"><div class="markdown-body">${renderedContent}</div></div>
      `;
    } catch (error) {
      console.error("Markdown parsing error:", error);
      renderedContent = content;
      messageDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">${renderedContent}</div>
      `;
    }
  } else {
    renderedContent = content;
    messageDiv.innerHTML = `
      <div class="message-avatar">${isUser ? "👤" : "🤖"}</div>
      <div class="message-content">${renderedContent}</div>
    `;
  }

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

/**
 * Get bot response from API
 * @param {string} userMessage - User's message
 */
async function getBotResponse(userMessage) {
  try {
    showTypingIndicator();

    const response = await fetch(`${API_BASE_URL}/response`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: userMessage }),
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
    console.error("Error:", error);
    addMessage(
      `抱歉，發生錯誤：${error.message}。請確認後端伺服器是否正在運行。`,
      false
    );
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
