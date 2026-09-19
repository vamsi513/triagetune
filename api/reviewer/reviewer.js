(() => {
  "use strict";

  let sessionKey = null;
  let suggestedCategory = null;

  const accessForm = document.querySelector("#access-form");
  const accessKey = document.querySelector("#access-key");
  const accessStatus = document.querySelector("#access-status");
  const suggestionForm = document.querySelector("#suggestion-form");
  const requestText = document.querySelector("#request-text");
  const characterCount = document.querySelector("#character-count");
  const suggestButton = document.querySelector("#suggest-button");
  const suggestionStatus = document.querySelector("#suggestion-status");
  const suggestedCategoryNode = document.querySelector("#suggested-category");
  const suggestedStatusNode = document.querySelector("#suggested-status");
  const reviewCategory = document.querySelector("#review-category");
  const acceptButton = document.querySelector("#accept-button");
  const overrideButton = document.querySelector("#override-button");
  const unsupportedButton = document.querySelector("#unsupported-button");
  const reviewStatus = document.querySelector("#review-status");

  function setStatus(node, message, kind = "") {
    node.textContent = message;
    node.className = `status ${kind}`.trim();
  }

  function authHeaders() {
    return {"X-TriageTune-Key": sessionKey};
  }

  function resetReview() {
    suggestedCategory = null;
    suggestedCategoryNode.textContent = "No suggestion available";
    suggestedStatusNode.textContent = "Pending input";
    reviewCategory.value = "";
    acceptButton.disabled = true;
    overrideButton.disabled = true;
    unsupportedButton.disabled = true;
    setStatus(reviewStatus, "Human review has not started.");
  }

  async function loadCategories() {
    const response = await fetch("/agent-assist/categories", {
      method: "GET",
      headers: authHeaders(),
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(response.status === 401 ? "The key was not accepted." : "Agent assist is unavailable.");
    }
    const body = await response.json();
    reviewCategory.replaceChildren(new Option("Choose a category", ""));
    for (const category of body.categories) {
      reviewCategory.add(new Option(category, category));
    }
  }

  accessForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const candidate = accessKey.value;
    setStatus(accessStatus, "Checking access…");
    sessionKey = candidate;
    try {
      await loadCategories();
      accessKey.value = "";
      requestText.disabled = false;
      suggestButton.disabled = false;
      reviewCategory.disabled = false;
      setStatus(accessStatus, "Unlocked for this browser tab.", "success");
      setStatus(suggestionStatus, "Ready for a synthetic or de-identified message.");
      requestText.focus();
    } catch (error) {
      sessionKey = null;
      requestText.disabled = true;
      suggestButton.disabled = true;
      reviewCategory.disabled = true;
      setStatus(accessStatus, error.message, "error");
    }
  });

  requestText.addEventListener("input", () => {
    characterCount.textContent = `${requestText.value.length} / 2000`;
  });

  suggestionForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!sessionKey) {
      setStatus(suggestionStatus, "Unlock the session first.", "error");
      return;
    }
    resetReview();
    suggestButton.disabled = true;
    setStatus(suggestionStatus, "Requesting a suggestion…");
    try {
      const response = await fetch("/agent-assist", {
        method: "POST",
        headers: {...authHeaders(), "Content-Type": "application/json"},
        body: JSON.stringify({text: requestText.value}),
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(response.status === 401 ? "Session access was rejected." : "Suggestion request failed.");
      }
      const body = await response.json();
      suggestedCategory = body.suggested_category;
      suggestedCategoryNode.textContent = suggestedCategory || "No valid category suggested";
      suggestedStatusNode.textContent = body.suggestion_status.replaceAll("_", " ");
      acceptButton.disabled = !suggestedCategory;
      overrideButton.disabled = false;
      unsupportedButton.disabled = false;
      if (suggestedCategory) {
        reviewCategory.value = suggestedCategory;
      }
      setStatus(suggestionStatus, "Suggestion returned. A human decision is required.", "success");
      setStatus(reviewStatus, "Pending human review.");
    } catch (error) {
      setStatus(suggestionStatus, error.message, "error");
    } finally {
      suggestButton.disabled = false;
    }
  });

  function completeLocalReview(message) {
    setStatus(reviewStatus, `${message} This decision was not sent or stored.`, "success");
    requestText.value = "";
    characterCount.textContent = "0 / 2000";
    acceptButton.disabled = true;
    overrideButton.disabled = true;
    unsupportedButton.disabled = true;
    suggestedCategory = null;
  }

  acceptButton.addEventListener("click", () => {
    if (suggestedCategory) {
      completeLocalReview(`Reviewer accepted “${suggestedCategory}”.`);
    }
  });

  overrideButton.addEventListener("click", () => {
    if (!reviewCategory.value) {
      setStatus(reviewStatus, "Choose a category before confirming.", "error");
      return;
    }
    completeLocalReview(`Reviewer selected “${reviewCategory.value}”.`);
  });

  unsupportedButton.addEventListener("click", () => {
    completeLocalReview("Reviewer marked the request unsupported.");
  });
})();
