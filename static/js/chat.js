// static/js/chat.js

const chatBox = document.getElementById("chatBox");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");

const decisionText = document.getElementById("decisionText");
const confidenceText = document.getElementById("confidenceText");
const topicText = document.getElementById("topicText");
const sourceText = document.getElementById("sourceText");
const debugText = document.getElementById("debugText");

function getUserId() {
  let userId = localStorage.getItem("rag_user_id");

  if (!userId) {
    userId = "web_user_" + Math.random().toString(36).slice(2, 10);
    localStorage.setItem("rag_user_id", userId);
  }

  return userId;
}

function appendMessage(role, text) {
  const wrapper = document.createElement("div");
  wrapper.className = "message " + role;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  wrapper.appendChild(bubble);
  chatBox.appendChild(wrapper);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function setInfo(data) {
  decisionText.textContent = data.decision || "-";
  confidenceText.textContent = data.confidence || "-";
  topicText.textContent = data.topic || "-";

  if (data.source) {
    sourceText.textContent = JSON.stringify(data.source, null, 2);
  } else {
    sourceText.textContent = data.source_type || "-";
  }

  debugText.textContent = JSON.stringify(data.debug || {}, null, 2);
}

async function askQuestion(question) {
  const payload = {
  user_id: getUserId(),
  question: question,
  candidate_k: 30,
  top_k: 1
};

  const response = await fetch("/api/ask", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText);
  }

  return await response.json();
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();

  if (!question) return;

  appendMessage("user", question);
  questionInput.value = "";
  sendBtn.disabled = true;
  sendBtn.textContent = "Đang trả lời...";

  try {
    const data = await askQuestion(question);

    appendMessage("bot", data.answer);
    setInfo(data);

    if (data.pending_id) {
      appendMessage(
        "bot",
        "Câu hỏi này đã được lưu vào hàng chờ xác minh. Pending ID: " + data.pending_id
      );
    }
  } catch (err) {
    appendMessage("bot", "Lỗi: " + err.message);
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = "Gửi";
    questionInput.focus();
  }
});