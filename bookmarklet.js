/**
 * bookmarklet.js
 *
 * 這不是要拿去執行的一般程式，是要轉成瀏覽器書籤用的。
 * 使用方式請看 README 或對話裡的說明。
 *
 * 功能：在 GitHub PR 頁面點這個書籤，會讀取目前頁面網址，
 * 呼叫本機的 PR review API (http://localhost:8000/review)，
 * 並在新分頁顯示 Markdown 格式的 review 報告。
 */
(function () {
  const prUrl = window.location.href.split("#")[0].split("?")[0];
  const isPrPage = /github\.com\/[^/]+\/[^/]+\/pull\/\d+/.test(prUrl);

  if (!isPrPage) {
    alert("請先切換到 GitHub 的 PR 頁面（網址要包含 /pull/數字），再點這個書籤。");
    return;
  }

  const resultWindow = window.open("", "_blank");
  resultWindow.document.write("<p style='font-family:monospace;padding:20px;'>審查中，請稍候...</p>");

  fetch("http://localhost:8000/review", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pr_url: prUrl }),
  })
    .then((res) => {
      if (!res.ok) {
        return res.json().then((err) => {
          throw new Error(err.detail || "API 回傳錯誤");
        });
      }
      return res.json();
    })
    .then((data) => {
      const report = data.markdown_report || JSON.stringify(data, null, 2);
      const escaped = report
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
      resultWindow.document.body.innerHTML =
        "<pre style='white-space:pre-wrap;font-family:monospace;padding:20px;line-height:1.6;'>" +
        escaped +
        "</pre>";
    })
    .catch((err) => {
      resultWindow.document.body.innerHTML =
        "<p style='font-family:monospace;padding:20px;color:red;'>呼叫失敗：" + err.message + "</p>" +
        "<p style='font-family:monospace;padding:0 20px;'>請確認本機的 uvicorn 服務是否有正常啟動（http://localhost:8000）。</p>";
    });
})();
