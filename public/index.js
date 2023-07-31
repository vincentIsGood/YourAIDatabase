
let lock = false;
let sources = new Set();
let jobId = "";
function queryAiDatabase(query){
    query = query.trim();
    if(query == "" || lock) return;

    lock = true;
    fetch("/aidb?query=" + encodeURIComponent(query))
    .then(res => res.text()).then(res =>{
        if(res == ""){
            createChatMessageDiv("Server returned empty response. Please keep waiting.", false);
            return;
        }
        jobId = res;

        const websocket = new WebSocket("ws://localhost:5023");
        // window.websocket = websocket;

        let assistantDiv = null;
        let currentMsgState = "";

        websocket.addEventListener("open", ()=>{
            websocket.send(JSON.stringify({"id": res}));
            createChatMessageDiv(query, true);
            assistantDiv = createChatMessageDiv("", false);
        });
        websocket.addEventListener("message", (event)=>{
            switch(event.data){
                case "[[[START]]]": 
                    document.querySelector("#stop")?.classList.remove("display-none");
                    currentMsgState = "response"; 
                break;
                case "[[[SOURCES]]]": 
                    currentMsgState = "sources";
                break;
                case "[[[END]]]": 
                    currentMsgState = "";
                    lock = false;
                    websocket.close();
                break;
                default:
                    if(currentMsgState == "response")
                        assistantDiv.textContent += event.data;
                    else if(currentMsgState == "sources"){
                        let sourceText = JSON.parse(event.data)["source"];
                        if(!sources.has(sourceText)){
                            sources.add(sourceText);
                            createSource(sourceText);
                        }
                    }
            }
        });
        websocket.addEventListener("close", (event)=>{
            document.querySelector("#stop")?.classList.add("display-none");
            lock = false;
        });
    });
}
function stopGeneration(){
    fetch("/aidb?id=" + encodeURIComponent(jobId), {
        method: "DELETE"
    });
}
function createChatMessageDiv(message, isUser = true){
    const divElement = document.createElement("div");
    const contentElement = document.createElement("div");
    if(isUser) divElement.classList.add("user");
    else divElement.classList.add("assistant");
    contentElement.classList.add("content");
    contentElement.textContent = message;
    divElement.appendChild(contentElement);
    document.querySelector(".conversation")?.appendChild(divElement);
    return contentElement;
}
function createSource(source){
    const sourceDiv = document.createElement("a");
    sourceDiv.classList.add("source");
    sourceDiv.textContent = source;
    sourceDiv.href = source;
    document.querySelector(".sources")?.appendChild(sourceDiv);
}


window.addEventListener("load", ()=>{
    queryHandlers();
    docUploadHandlers();
});

function queryHandlers(){
    const userInputField = document.querySelector("#user-input");
    const queryButton = document.querySelector("#query");
    const stopButton = document.querySelector("#stop");

    userInputField.addEventListener("keydown", (e)=>{
        if(e.key == "Enter"){
            queryAiDatabase(userInputField.value);
            userInputField.value = "";
            queryButton.classList.add("display-none");
        }
    });
    userInputField.addEventListener("input", ()=>{
        if(userInputField.value.length > 0)
            queryButton.classList.remove("display-none");
        else queryButton.classList.add("display-none");
    });

    queryButton.addEventListener("click", ()=>{
        queryAiDatabase(userInputField.value);
        userInputField.value = "";
        queryButton.classList.add("display-none");
    });
    
    stopButton.addEventListener("click", ()=>{
        stopGeneration();
    });
}

function docUploadHandlers(){
    const dragNDropUploadField = document.querySelector(".drag-n-drop");
    const dragNDropStatusMsg = document.querySelector(".drag-n-drop .status");
    const addIcon = document.querySelector(".add-icon");
    const uploadIcon = document.querySelector(".upload-icon");
    
    dragNDropUploadField.addEventListener("drop", async (e)=>{
        e.preventDefault();
        for(let droppedItem of e.dataTransfer.items){
            if(droppedItem.kind == "file"){
                const file = droppedItem.getAsFile();
                await uploadFile(file);
            }
        }
        setTimeout(()=>{
            addIcon.classList.remove("display-none");
            uploadIcon.classList.add("display-none");
            dragNDropStatusMsg.textContent = "Drop files here";
        }, 1000);
    });
    dragNDropUploadField.addEventListener("dragover", (e)=>{
        e.preventDefault();
        addIcon.classList.add("display-none");
        uploadIcon.classList.remove("display-none");
        dragNDropStatusMsg.textContent = "Upload";
    });
    dragNDropUploadField.addEventListener("dragleave", (e)=>{
        e.preventDefault();
        addIcon.classList.remove("display-none");
        uploadIcon.classList.add("display-none");
        dragNDropStatusMsg.textContent = "Drop files here";
    });

    addIcon.addEventListener("click", selectFileToUpload);
    uploadIcon.addEventListener("click", selectFileToUpload);
}

let previousInputFile = null;
function selectFileToUpload(){
    if(previousInputFile){
        previousInputFile.remove();
    }

    const inputFile = document.createElement("input");
    inputFile.type = "file";
    inputFile.addEventListener("change", (e)=>{
        const file = inputFile.files[0];
        uploadFile(file);
    });
    inputFile.click();
    previousInputFile = inputFile;
}

/**
 * @param {File} file 
 */
async function uploadFile(file){
    const dragNDropStatusMsg = document.querySelector(".drag-n-drop .status");
    dragNDropStatusMsg.textContent = "Uploading...";

    fetch("/aidb/upload?name=" + file.name, {
        method: "POST", 
        body: await file.arrayBuffer(),
        headers: {
            "content-type": file.type
        }
    }).then(()=>{
        dragNDropStatusMsg.textContent = "Uploaded";
    }).catch(()=>{
        dragNDropStatusMsg.textContent = "Error Encountered";
    });
}