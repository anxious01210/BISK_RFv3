// sort_faces_terminal.js
function showFinalScriptMessage(message) {
    const terminal = document.getElementById("sortFacesTerminal");
    const statusText = document.getElementById("sortFacesStatusText");
    terminal.innerText = message;
    statusText.innerText = message;
}








// function openTerminalModal(jobId) {
//     if (document.getElementById("terminalModal")) {
//         document.getElementById("terminalModal").remove();
//     }
//
//     const modalHTML = `
//     <div id="terminalModal" class="modal-overlay" style="display:flex; align-items:center; justify-content:center;">
//         <div class="modal-content" style="width: 800px; height: 500px; background: #111; color: #eee; padding: 16px; border-radius: 8px; display: flex; flex-direction: column;">
//             <h3 style="margin-top: 0;">🧠 Running Sort Faces Script</h3>
//             <div id="terminalOutput" style="flex:1; overflow-y:auto; background:#000; padding:10px; font-family:monospace; font-size:13px; white-space:pre-wrap; border-radius:4px; border:1px solid #555;"></div>
//             <div style="margin-top:10px; display:flex; justify-content:space-between;">
//                 <button onclick="stopSortFacesJob('${jobId}')">🛑 Stop</button>
//                 <button onclick="document.getElementById('terminalModal').remove()">❌ Close</button>
//             </div>
//         </div>
//     </div>`;
//
//     document.body.insertAdjacentHTML("beforeend", modalHTML);
//     streamSortFacesOutput(jobId);
// }
//
// function streamSortFacesOutput(jobId) {
//     const terminal = document.getElementById("terminalOutput");
//     const evtSource = new EventSource(`/file-manager/stream-sort-faces/${jobId}/`);
//
//     evtSource.onmessage = function (e) {
//         terminal.textContent += e.data + "\n";
//         terminal.scrollTop = terminal.scrollHeight;
//     };
//
//     evtSource.onerror = function () {
//         terminal.textContent += "\n❌ Connection closed or script completed.\n";
//         evtSource.close();
//     };
// }
//
// function stopSortFacesJob(jobId) {
//     fetch(`/file-manager/stop-sort-faces/${jobId}/`, {
//         method: 'POST',
//         headers: { 'X-CSRFToken': getCSRFToken() },
//     })
//     .then(() => {
//         document.getElementById("terminalOutput").textContent += "\n🛑 Job stopped by user.\n";
//     })
//     .catch(() => {
//         alert("Failed to stop the job.");
//     });
// }
//
// function getCSRFToken() {
//     return document.cookie.split('; ').find(row => row.startsWith('csrftoken'))?.split('=')[1];
// }