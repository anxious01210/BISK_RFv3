document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
        selectedItems = [];
        updateToolbarVisibility();
        document.querySelectorAll(".file-item").forEach(i => i.classList.remove("selected"));
        updateInfoPanel();
    }
});

document.getElementById("refreshBtn").addEventListener("click", () => {
    loadFolder(currentPath);
});

document.getElementById("limitSelect").addEventListener("change", (e) => {
    fileLimit = e.target.value;
    loadFolder(currentPath);
});

document.getElementById("viewToggle").addEventListener("change", (e) => {
    const view = e.target.value;
    const container = document.getElementById("folderContents");
    if (view === "list") {
        container.classList.remove("grid-view");
        container.classList.add("list-view");
    } else {
        container.classList.remove("list-view");
        container.classList.add("grid-view");
    }
});