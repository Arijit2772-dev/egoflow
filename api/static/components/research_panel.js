window.renderResearchPanel = function renderResearchPanel(papers) {
  const list = document.getElementById("research-list");
  list.innerHTML = "";
  papers.forEach((paper) => {
    const link = document.createElement("a");
    link.href = paper.link;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = paper.citation;
    list.appendChild(link);
  });
  document.getElementById("research-toggle").addEventListener("click", () => {
    list.classList.toggle("open");
  });
};
