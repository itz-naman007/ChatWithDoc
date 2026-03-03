// Upload Files
async function uploadFiles() {

    const fileInput = document.getElementById("file-input");
    const status = document.getElementById("upload-status");

    if (fileInput.files.length === 0) {
        status.innerText = "Please select files first.";
        return;
    }

    const formData = new FormData();

    for (let file of fileInput.files) {
        formData.append("file", file);
    }

    status.innerText = "Processing documents...";

    const response = await fetch("/upload", {
        method: "POST",
        body: formData
    });

    const data = await response.json();
    status.innerText = data.status;
}


// Send Chat Message
async function sendMessage() {

    const inputField = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const message = inputField.value.trim();

    if (message === "") return;

    chatBox.innerHTML += `
        <div class="flex justify-end">
            <div class="bg-blue-600 px-4 py-2 rounded-2xl max-w-md text-sm">
                ${message}
            </div>
        </div>
    `;

    inputField.value = "";
    chatBox.scrollTop = chatBox.scrollHeight;

    const loadingId = "loading-" + Date.now();
    chatBox.innerHTML += `
        <div id="${loadingId}" class="flex justify-start">
            <div class="bg-slate-700 px-4 py-2 rounded-2xl max-w-md text-sm animate-pulse">
                Thinking...
            </div>
        </div>
    `;

    chatBox.scrollTop = chatBox.scrollHeight;

    const selectedMode = document.getElementById("mode-select").value;

    const response = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            question: message,
            mode: selectedMode
        })
    });

    const data = await response.json();

    document.getElementById(loadingId).remove();

    chatBox.innerHTML += `
        <div class="flex justify-start">
            <div class="bg-slate-700 px-4 py-2 rounded-2xl max-w-md text-sm">
                ${data.answer}
            </div>
        </div>
    `;

    chatBox.scrollTop = chatBox.scrollHeight;
}


// Enter key support
document.getElementById("user-input")
    .addEventListener("keypress", function(e) {
        if (e.key === "Enter") sendMessage();
    });
