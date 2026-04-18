window.attachFieldTooltips = function attachFieldTooltips(papers) {
  const tooltip = document.getElementById("tooltip");
  const byField = new Map();
  papers.forEach((paper) => {
    paper.contributes.forEach((field) => byField.set(field, paper.citation));
  });
  document.querySelectorAll("[data-field]").forEach((node) => {
    node.addEventListener("mouseenter", (event) => {
      const field = event.currentTarget.getAttribute("data-field");
      tooltip.textContent = byField.get(field) || byField.get(field.split(".")[0]) || "EgoFlow schema field";
      tooltip.hidden = false;
    });
    node.addEventListener("mousemove", (event) => {
      tooltip.style.left = `${event.clientX + 12}px`;
      tooltip.style.top = `${event.clientY + 12}px`;
    });
    node.addEventListener("mouseleave", () => {
      tooltip.hidden = true;
    });
  });
};
