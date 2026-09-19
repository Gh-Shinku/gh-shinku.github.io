(() => {
  const mobileQuery = window.matchMedia("(max-width: 1024px)");
  const toc = document.querySelector(".post-toc");

  if (toc) {
    const toggle = toc.querySelector(".toc-toggle");
    const icon = toc.querySelector(".toc-toggle-icon");
    const panel = toc.querySelector(".toc-panel");
    const links = [...toc.querySelectorAll('#TableOfContents a[href^="#"]')];

    if (!links.length) {
      toc.hidden = true;
    } else {
      const setExpanded = (expanded) => {
        toggle.setAttribute("aria-expanded", String(expanded));
        panel.hidden = !expanded;
        icon.textContent = expanded ? "−" : "+";
      };

      setExpanded(!mobileQuery.matches);
      toggle.addEventListener("click", () => {
        setExpanded(toggle.getAttribute("aria-expanded") !== "true");
      });
      mobileQuery.addEventListener("change", (event) => setExpanded(!event.matches));

      const trackedHeadings = links
        .map((link) => {
          const id = decodeURIComponent(link.hash.slice(1));
          return { heading: document.getElementById(id), link };
        })
        .filter(({ heading }) => heading);

      let ticking = false;
      const updateActiveHeading = () => {
        let active = trackedHeadings[0];
        for (const item of trackedHeadings) {
          if (item.heading.getBoundingClientRect().top <= 120) active = item;
          else break;
        }

        for (const item of trackedHeadings) {
          const isActive = item === active;
          item.link.classList.toggle("is-active", isActive);
          if (isActive) item.link.setAttribute("aria-current", "location");
          else item.link.removeAttribute("aria-current");
        }
        ticking = false;
      };

      window.addEventListener("scroll", () => {
        if (!ticking) {
          window.requestAnimationFrame(updateActiveHeading);
          ticking = true;
        }
      }, { passive: true });
      updateActiveHeading();

      links.forEach((link) => link.addEventListener("click", () => {
        if (mobileQuery.matches) setExpanded(false);
      }));
    }
  }

  const copyText = async (text) => {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return;
    }

    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.append(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  };

  document.querySelectorAll(".post-content .highlight").forEach((highlight) => {
    const code = highlight.querySelector("code");
    if (!code) return;

    const toolbar = document.createElement("div");
    toolbar.className = "code-toolbar";

    const language = document.createElement("span");
    language.className = "code-language";
    language.textContent = (code.dataset.lang || "text").toUpperCase();

    const copyButton = document.createElement("button");
    copyButton.className = "code-copy";
    copyButton.type = "button";
    copyButton.textContent = "复制";
    copyButton.setAttribute("aria-label", "复制代码");

    copyButton.addEventListener("click", async () => {
      try {
        await copyText(code.innerText);
        copyButton.textContent = "已复制";
      } catch {
        copyButton.textContent = "复制失败";
      }
      window.setTimeout(() => { copyButton.textContent = "复制"; }, 1500);
    });

    toolbar.append(language, copyButton);
    highlight.prepend(toolbar);
  });

  document.querySelectorAll(".post-content table").forEach((table) => {
    if (table.parentElement.classList.contains("table-scroll")) return;
    const wrapper = document.createElement("div");
    wrapper.className = "table-scroll";
    wrapper.tabIndex = 0;
    wrapper.setAttribute("role", "region");
    wrapper.setAttribute("aria-label", "可横向滚动的表格");
    table.parentNode.insertBefore(wrapper, table);
    wrapper.append(table);
  });

  const dialog = document.querySelector(".image-lightbox");
  if (dialog && typeof dialog.showModal === "function") {
    const preview = dialog.querySelector("img");
    const caption = dialog.querySelector(".image-lightbox-caption");
    const closeButton = dialog.querySelector(".image-lightbox-close");

    const openImage = (image) => {
      preview.src = image.currentSrc || image.src;
      preview.alt = image.alt;
      caption.textContent = image.alt;
      caption.hidden = !image.alt;
      dialog.showModal();
    };

    document.querySelectorAll(".post-content img").forEach((image) => {
      const paragraph = image.closest("p");
      if (image.alt && paragraph?.parentElement?.classList.contains("post-content") &&
          !paragraph.textContent.trim() && paragraph.querySelectorAll("img").length === 1) {
        const figure = document.createElement("figure");
        figure.className = "content-figure";
        paragraph.replaceWith(figure);
        while (paragraph.firstChild) figure.append(paragraph.firstChild);
        const figcaption = document.createElement("figcaption");
        figcaption.textContent = image.alt;
        figure.append(figcaption);
      }

      image.classList.add("zoomable-image");
      image.tabIndex = 0;
      image.setAttribute("role", "button");
      image.setAttribute("aria-label", image.alt ? `放大图片：${image.alt}` : "放大图片");
      image.addEventListener("click", (event) => {
        event.preventDefault();
        openImage(image);
      });
      image.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openImage(image);
        }
      });
    });

    closeButton.addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close();
    });
  }
})();
