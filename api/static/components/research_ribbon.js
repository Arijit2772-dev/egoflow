window.renderResearchRibbon = function renderResearchRibbon(papers) {
  const ribbon = document.getElementById("research-ribbon");
  ribbon.innerHTML = "";
  const label = document.createElement("strong");
  label.textContent = "Research-backed:";
  ribbon.appendChild(label);
  papers.forEach((paper) => {
    const link = document.createElement("a");
    const shortName = paper.citation.split(" - ")[0];
    link.href = paper.link;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = shortName;
    ribbon.appendChild(link);
  });
};
