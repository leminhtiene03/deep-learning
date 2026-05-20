// static/js/admin.js

const adminTokenInput = document.getElementById("adminToken");
const saveTokenBtn = document.getElementById("saveTokenBtn");

const loadPendingBtn = document.getElementById("loadPendingBtn");
const loadCorrectionsBtn = document.getElementById("loadCorrectionsBtn");

const pendingList = document.getElementById("pendingList");
const correctionList = document.getElementById("correctionList");

function getAdminToken() {
  return localStorage.getItem("admin_token") || "local-admin-token";
}

function setAdminToken(token) {
  localStorage.setItem("admin_token", token);
}

adminTokenInput.value = getAdminToken();

saveTokenBtn.addEventListener("click", () => {
  setAdminToken(adminTokenInput.value.trim());
  alert("Đã lưu admin token");
});

async function apiGet(url) {
  const response = await fetch(url, {
    headers: {
      "X-Admin-Token": getAdminToken()
    }
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return await response.json();
}

async function apiPost(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Admin-Token": getAdminToken()
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return await response.json();
}

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function loadPending() {
  pendingList.innerHTML = "Đang tải...";

  try {
    const data = await apiGet("/api/admin/pending?status=pending&limit=50");

    if (!data.items.length) {
      pendingList.innerHTML = "<p>Không có pending question.</p>";
      return;
    }

    pendingList.innerHTML = "";

    data.items.forEach((item) => {
      const card = document.createElement("div");
      card.className = "item-card";

      const teacherQuestions = item.teacher_questions || [];

      card.innerHTML = `
        <h3>Pending #${item.id}</h3>
        <div class="meta">
          User: ${escapeHtml(item.user_id)} |
          Topic: ${escapeHtml(item.topic)} |
          Created: ${escapeHtml(item.created_at)}
        </div>

        <p><b>Question:</b></p>
        <p>${escapeHtml(item.question)}</p>

        <p><b>Reason:</b></p>
        <p>${escapeHtml(item.reason)}</p>

        <p><b>Suggested teacher questions:</b></p>
        <pre>${escapeHtml(teacherQuestions.join("\n"))}</pre>

        <p><b>Context snapshot:</b></p>
        <pre>${escapeHtml((item.context_snapshot || "").slice(0, 1000))}</pre>

        <textarea placeholder="Nhập câu trả lời đã xác nhận từ thầy cô/admin..."></textarea>

        <button class="success answer-btn">Lưu câu trả lời</button>
      `;

      const textarea = card.querySelector("textarea");
      const answerBtn = card.querySelector(".answer-btn");

      answerBtn.addEventListener("click", async () => {
        const answer = textarea.value.trim();

        if (!answer) {
          alert("Bạn chưa nhập câu trả lời.");
          return;
        }

        answerBtn.disabled = true;
        answerBtn.textContent = "Đang lưu...";

        try {
          const result = await apiPost("/api/admin/answer-pending", {
            pending_id: item.id,
            answer: answer,
            verified_by: "web_admin",
            source: "teacher"
          });

          alert(
            "Đã lưu confirmed answer #" +
            result.confirmed_answer_id +
            "\nCorrections created: " +
            result.correction_count
          );

          await loadPending();
          await loadCorrections();
        } catch (err) {
          alert("Lỗi: " + err.message);
        } finally {
          answerBtn.disabled = false;
          answerBtn.textContent = "Lưu câu trả lời";
        }
      });

      pendingList.appendChild(card);
    });
  } catch (err) {
    pendingList.innerHTML = "<p>Lỗi: " + escapeHtml(err.message) + "</p>";
  }
}

async function loadCorrections() {
  correctionList.innerHTML = "Đang tải...";

  try {
    const data = await apiGet("/api/admin/corrections?status=pending&limit=50");

    if (!data.items.length) {
      correctionList.innerHTML = "<p>Không có correction pending.</p>";
      return;
    }

    correctionList.innerHTML = "";

    data.items.forEach((item) => {
      const card = document.createElement("div");
      card.className = "item-card";

      card.innerHTML = `
        <h3>Correction #${item.id}</h3>
        <div class="meta">
          Answer log: ${escapeHtml(item.answer_log_id)} |
          Status: ${escapeHtml(item.status)}
        </div>

        <p><b>Reason:</b></p>
        <p>${escapeHtml(item.reason)}</p>

        <p><b>Old answer:</b></p>
        <pre>${escapeHtml((item.old_answer || "").slice(0, 1000))}</pre>

        <p><b>New answer:</b></p>
        <pre>${escapeHtml((item.new_answer || "").slice(0, 1000))}</pre>

        <button class="success mark-btn">Đánh dấu đã thông báo</button>
      `;

      const markBtn = card.querySelector(".mark-btn");

      markBtn.addEventListener("click", async () => {
        markBtn.disabled = true;
        markBtn.textContent = "Đang cập nhật...";

        try {
          await apiPost("/api/admin/corrections/mark-notified", {
            correction_id: item.id
          });

          alert("Đã đánh dấu notified.");
          await loadCorrections();
        } catch (err) {
          alert("Lỗi: " + err.message);
        } finally {
          markBtn.disabled = false;
          markBtn.textContent = "Đánh dấu đã thông báo";
        }
      });

      correctionList.appendChild(card);
    });
  } catch (err) {
    correctionList.innerHTML = "<p>Lỗi: " + escapeHtml(err.message) + "</p>";
  }
}

loadPendingBtn.addEventListener("click", loadPending);
loadCorrectionsBtn.addEventListener("click", loadCorrections);

loadPending();
loadCorrections();