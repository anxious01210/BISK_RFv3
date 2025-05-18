
document.addEventListener("DOMContentLoaded", () => {
    const currentFolderPath = decodeURIComponent(location.pathname.replace('/file-manager/', ''));
    loadFolder(currentFolderPath);
});