// Generate or retrieve session ID
let sessionId = localStorage.getItem('amenify_session_id');
if (!sessionId) {
    sessionId = 'session_' + Math.random().toString(36).substring(2, 15);
    localStorage.setItem('amenify_session_id', sessionId);
}

const chatBox = document.getElementById('chat-box');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const typingIndicator = document.getElementById('typing-indicator');

function appendMessage(role, text) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', role);
    msgDiv.innerText = text;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return msgDiv;
}

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    // Append user message
    appendMessage('user', text);
    chatInput.value = '';
    
    // UI state
    sendBtn.disabled = true;
    chatInput.disabled = true;
    typingIndicator.style.display = 'flex';
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, message: text })
        });

        typingIndicator.style.display = 'none';

        if (!response.ok) {
            appendMessage('assistant', 'Sorry, I encountered an error. Please try again later.');
            return;
        }

        // Handle streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let done = false;
        
        // Create an empty assistant message box to fill
        const assistantMsgDiv = appendMessage('assistant', '');

        while (!done) {
            const { value, done: readerDone } = await reader.read();
            done = readerDone;
            if (value) {
                const chunk = decoder.decode(value, { stream: true });
                assistantMsgDiv.innerText += chunk;
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        }
    } catch (err) {
        console.error(err);
        typingIndicator.style.display = 'none';
        appendMessage('assistant', 'Network error. Please check your connection.');
    } finally {
        sendBtn.disabled = false;
        chatInput.disabled = false;
        chatInput.focus();
    }
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});
