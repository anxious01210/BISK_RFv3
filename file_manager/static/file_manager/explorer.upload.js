let pendingFolderFiles = [];
let lastUploadMode = "preserve";

function showRetryDialog() {
    const resultBox = document.getElementById("uploadResultMessage");
    const resultText = document.getElementById("uploadResultText");

    resultText.innerHTML = `❌ Upload cancelled.<br><button id="retryUploadBtn" style="margin-top:6px; padding:4px 10px; background:#2196f3; color:white; border:none; border-radius:4px;">🔁 Retry</button>`;
    resultBox.style.display = "flex";

    document.getElementById("retryUploadBtn").onclick = () => {
        resultBox.style.display = "none";
        handleFolderUpload(lastUploadMode);
    };
}

document.getElementById("uploadInput").addEventListener("change", async function () {
    const files = this.files;
    if (!files.length) return;

    const formData = new FormData();
    for (let file of files) {
        formData.append("files", file);
    }
    formData.append("path", currentPath);
    formData.append("folder_mode", "flat");
    formData.append("relative_paths", JSON.stringify([...files].map(f => f.name)));

    try {
        const res = await fetch("/file-manager/upload/", {
            method: "POST",
            body: formData
        });

        const result = await res.json();
        const resultBox = document.getElementById("uploadResultMessage");
        const resultText = document.getElementById("uploadResultText");

        if (result.status === "ok") {
            resultText.innerText = `✅ Uploaded ${result.saved.length} file(s).`;
        } else {
            resultText.innerText = "⚠️ Upload failed: " + (result.message || "Unknown error");
        }

        resultBox.style.display = "flex";
        setTimeout(() => resultBox.style.display = "none", 3000);
        loadFolder(currentPath);
    } catch (err) {
        console.error("Upload error", err);
        alert("❌ Upload error: " + err.message);
    }

    this.value = "";
});

document.getElementById("folderUploadInput").addEventListener("change", function () {
    if (this.files.length > 0) {
        pendingFolderFiles = [...this.files];
        document.getElementById("uploadModeModal").style.display = "flex";
    }
    this.value = "";
});

document.getElementById("uploadFlat").onclick = () => handleFolderUpload("flat");
document.getElementById("uploadPreserve").onclick = () => handleFolderUpload("preserve");

async function handleFolderUpload(mode) {
    lastUploadMode = mode;

    const files = pendingFolderFiles;
    document.getElementById("uploadModeModal").style.display = "none";
    if (!files || !files.length) return;

    const formData = new FormData();
    const relativePaths = [];

    for (let file of files) {
        formData.append("files", file);
        relativePaths.push(file.webkitRelativePath);
    }

    formData.append("path", currentPath);
    formData.append("folder_mode", mode);
    formData.append("relative_paths", JSON.stringify(relativePaths));

    const spinner = document.getElementById("uploadSpinner");
    document.getElementById("uploadSpinnerMessage").innerText = "Uploading " + files.length + " file(s)...";
    spinner.style.display = "flex";

    try {
        const xhr = new XMLHttpRequest();
        let uploadStart = Date.now();

        xhr.open("POST", "/file-manager/upload/", true);

        xhr.upload.onprogress = function (e) {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                const progressEl = document.getElementById("uploadProgressBar");

                progressEl.value = percent;
                document.getElementById("uploadSpinnerMessage").innerText = `Uploading (${percent}%)`;

                progressEl.classList.remove("low", "mid", "high");
                if (percent < 33) progressEl.classList.add("low");
                else if (percent < 66) progressEl.classList.add("mid");
                else progressEl.classList.add("high");

                const elapsed = (Date.now() - uploadStart) / 1000;
                const speed = e.loaded / elapsed;
                const remaining = e.total - e.loaded;
                const eta = speed > 0 ? Math.ceil(remaining / speed) : 0;

                document.getElementById("uploadETA").innerText = `ETA: ${formatETA(eta)}`;
            }
        };

        xhr.onload = function () {
            spinner.style.display = "none";

            const resultBox = document.getElementById("uploadResultMessage");
            const resultText = document.getElementById("uploadResultText");

            let result;
            try {
                result = JSON.parse(xhr.responseText);
            } catch (err) {
                resultText.innerText = "❌ Invalid server response.";
                resultBox.style.display = "flex";
                return;
            }

            if (xhr.status === 200 && result.status === "ok") {
                const numFiles = result.saved?.length || 0;
                const folderSet = new Set(result.saved.map(path => path.split("/").slice(0, -1).join("/")));
                const numFolders = folderSet.size;
                resultText.innerText = `✅ Uploaded ${numFiles} file(s) into ${numFolders} folder(s).`;
            } else {
                resultText.innerText = "⚠️ Upload failed: " + (result.message || "Unknown error");
            }

            resultBox.style.display = "flex";
            setTimeout(() => resultBox.style.display = "none", 3500);
            loadFolder(currentPath);
        };

        xhr.onerror = function () {
            spinner.style.display = "none";
            alert("❌ Upload failed due to a network error.");
        };

        const cancelBtn = document.getElementById("uploadCancelBtn");
        cancelBtn.onclick = () => {
            xhr.abort();
            spinner.style.display = "none";
            showRetryDialog();
        };

        document.getElementById("uploadProgressBar").value = 0;
        document.getElementById("uploadETA").innerText = "ETA: —";
        uploadStart = Date.now();

        xhr.send(formData);
    } catch (err) {
        spinner.style.display = "none";
        alert("❌ Upload error: " + err.message);
    }

    pendingFolderFiles = null;
}