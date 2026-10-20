/**
 * bookmarklet.js
 *
 * This is not an ordinary program to run; it is meant to be turned into a browser bookmark.
 * See the README (or the explanation given in the chat) for how to use it.
 *
 * What it does: clicking this bookmark on a GitHub PR page reads the current page URL,
 * calls the local PR review API (http://localhost:8000/review),
 * and shows the review report in Markdown format in a new tab.
 */
(function () {
  const prUrl = window.location.href.split("#")[0].split("?")[0];
  const isPrPage = /github\.com\/[^/]+\/[^/]+\/pull\/\d+/.test(prUrl);

  if (!isPrPage) {
    alert("Please switch to a GitHub PR page (the URL must contain /pull/<number>) first, then click this bookmark.");
    return;
  }

  const resultWindow = window.open("", "_blank");
  resultWindow.document.write("<p style='font-family:monospace;padding:20px;'>Reviewing, please wait...</p>");

  fetch("http://localhost:8000/review", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pr_url: prUrl }),
  })
    .then((res) => {
      if (!res.ok) {
        return res.json().then((err) => {
          throw new Error(err.detail || "API returned an error");
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
        "<p style='font-family:monospace;padding:20px;color:red;'>Request failed: " + err.message + "</p>" +
        "<p style='font-family:monospace;padding:0 20px;'>Please check that the local uvicorn service is running (http://localhost:8000).</p>";
    });
})();
