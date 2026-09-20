const messages = document.querySelector("#messages");
const form = document.querySelector("#chatForm");
const input = document.querySelector("#query");
const clearBtn = document.querySelector("#clearBtn");
const conversationId = "web-" + Math.random().toString(36).slice(2);

function addMessage(role, text, result) {
  const item = document.createElement("div");
  item.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  item.appendChild(bubble);
  if (result) {
    const meta = document.createElement("div");
    meta.className = "meta";
    if (typeof result.confidence === "number") {
      const confidence = document.createElement("span");
      confidence.className = "chip confidence";
      confidence.textContent = `${Math.round(result.confidence * 100)}% grounded`;
      meta.appendChild(confidence);
    }
    (result.sources || []).slice(0, 4).forEach(source => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = `Source · ${source.id}`;
      meta.appendChild(chip);
    });
    if (result.escalation) {
      const escalation = document.createElement("span");
      escalation.className = "chip escalated";
      escalation.textContent = "Human review recommended";
      meta.appendChild(escalation);
    }
    item.appendChild(meta);
  }
  messages.appendChild(item);
  item.scrollIntoView({behavior:"smooth", block:"nearest"});
}

async function ask(question) {
  const text = question.trim();
  if (!text) return;
  document.querySelector(".welcome")?.remove();
  document.querySelector(".suggestions")?.remove();
  addMessage("user", text);
  input.value = "";
  const item = document.createElement("div");
  item.className = "message assistant";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = "Searching trusted guidance…";
  item.appendChild(bubble);
  messages.appendChild(item);
  try {
    const payload = {query:text, conversation_id:conversationId};
    const response = await fetch("/query/stream", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
    // Allow the UI to work with an older server process while it is restarted.
    if (response.status === 404) {
      const fallback = await fetch("/query", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
      const result = await fallback.json();
      item.remove();
      addMessage("assistant", result.answer || "I could not generate a response.", result);
      return;
    }
    if (!response.ok || !response.body) throw new Error("stream unavailable");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let result = null;
    bubble.textContent = "";
    while (true) {
      const part = await reader.read();
      if (part.done) break;
      buffer += decoder.decode(part.value, {stream:true});
      const events = buffer.split("\n\n");
      buffer = events.pop();
      for (const event of events) {
        if (!event.startsWith("data: ")) continue;
        const payload = JSON.parse(event.slice(6));
        if (payload.type === "token") {
          bubble.textContent += payload.text;
          item.scrollIntoView({behavior:"smooth", block:"nearest"});
        } else if (payload.type === "done") {
          result = payload.result;
        }
      }
    }
    if (result) {
      item.remove();
      addMessage("assistant", result.answer, result);
    }
  } catch (error) {
    item.remove();
    addMessage("assistant", "The support service is temporarily unavailable. Please try again.");
  }
}

form.addEventListener("submit", event => { event.preventDefault(); ask(input.value); });
input.addEventListener("keydown", event => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); ask(input.value); } });
document.querySelectorAll("[data-query]").forEach(button => button.addEventListener("click", () => ask(button.dataset.query)));
clearBtn.addEventListener("click", () => window.location.reload());
