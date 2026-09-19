(() => {
  const form = document.querySelector(".search-form");
  if (!form) return;

  const input = form.querySelector('input[type="search"]');
  const status = document.querySelector(".search-status");
  const results = document.querySelector(".search-results");
  let articles = [];
  let debounceTimer;

  const normalize = (value) => String(value || "").toLocaleLowerCase().trim();

  const articleScore = (article, terms) => {
    const title = normalize(article.title);
    const tags = normalize((article.tags || []).join(" "));
    const summary = normalize(article.summary);
    const content = normalize(article.content);
    let score = 0;

    for (const term of terms) {
      if (!title.includes(term) && !tags.includes(term) &&
          !summary.includes(term) && !content.includes(term)) return -1;
      if (title === term) score += 120;
      else if (title.startsWith(term)) score += 70;
      else if (title.includes(term)) score += 45;
      if (tags.includes(term)) score += 30;
      if (summary.includes(term)) score += 12;
      if (content.includes(term)) score += 3;
    }
    return score;
  };

  const renderResult = (article) => {
    const item = document.createElement("li");
    const result = document.createElement("article");
    const title = document.createElement("h2");
    const link = document.createElement("a");
    const meta = document.createElement("p");
    const summary = document.createElement("p");

    result.className = "search-result";
    link.href = article.url;
    link.textContent = article.title;
    title.append(link);
    meta.className = "search-result-meta";
    meta.textContent = [article.date, ...(article.tags || [])].filter(Boolean).join(" · ");
    summary.className = "search-result-summary";
    summary.textContent = article.summary;
    result.append(title, meta, summary);
    item.append(result);
    return item;
  };

  const search = () => {
    const query = input.value.trim();
    const url = new URL(window.location.href);
    if (query) url.searchParams.set("q", query);
    else url.searchParams.delete("q");
    window.history.replaceState({}, "", url);
    results.replaceChildren();

    if (!query) {
      status.textContent = "Enter a keyword to search the articles.";
      return;
    }

    const terms = normalize(query).split(/\s+/).filter(Boolean);
    const matches = articles
      .map((article) => ({ article, score: articleScore(article, terms) }))
      .filter(({ score }) => score >= 0)
      .sort((left, right) => right.score - left.score ||
        String(right.article.date).localeCompare(String(left.article.date)))
      .slice(0, 50);

    status.textContent = matches.length === 0
      ? "No articles found."
      : `${matches.length} ${matches.length === 1 ? "article" : "articles"} found.`;
    results.append(...matches.map(({ article }) => renderResult(article)));
  };

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    search();
  });
  input.addEventListener("input", () => {
    window.clearTimeout(debounceTimer);
    debounceTimer = window.setTimeout(search, 120);
  });

  const initialQuery = new URL(window.location.href).searchParams.get("q") || "";
  input.value = initialQuery;
  fetch(form.dataset.indexUrl)
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((data) => {
      articles = Array.isArray(data) ? data : [];
      search();
    })
    .catch(() => {
      status.textContent = "The search index could not be loaded.";
      input.disabled = true;
    });
})();
