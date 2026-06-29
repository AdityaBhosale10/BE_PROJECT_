import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Send,
  ArrowUpRight,
  X,
  Plus,
  MessageSquare,
  Trash2,
  PanelLeftClose,
  Menu,
  ArrowLeft,
  Image as ImageIcon,
  Loader2,
} from "lucide-react";
import "./App.css";

const STORAGE_KEY = "productgpt_conversations_v1";

function createId() {
  return crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function loadStore() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { activeId: null, chats: [] };
    const parsed = JSON.parse(raw);
    return {
      activeId: parsed.activeId ?? null,
      chats: Array.isArray(parsed.chats) ? parsed.chats : [],
    };
  } catch {
    return { activeId: null, chats: [] };
  }
}

function saveStore(store) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
}

function titleFromMessage(text) {
  const t = (text || "").trim();
  if (!t) return "New chat";
  return t.length > 42 ? `${t.slice(0, 42)}…` : t;
}

function normalizeProduct(p) {
  return {
    title: p.title ?? p.name ?? "Product",
    description: p.description ?? p.reason ?? "",
    image: p.image ?? p.image_url ?? "",
    url: p.url ?? p.source_url ?? "",
    price: p.price ?? null,
    source: p.source ?? null,
  };
}

function responseToMessages(response) {
  const out = [];
  if (response.default) {
    out.push({ text: response.default, sender: "bot", streaming: false });
  }
  if (response.initial?.message) {
    out.push({
      text: String(response.initial.message).replace(/^-+\s*/, ""),
      sender: "bot",
      streaming: false,
    });
  }
  if (response.products?.length) {
    out.push({ sender: "products", items: response.products.map(normalizeProduct) });
  }
  if (response.final?.message) {
    out.push({
      text: String(response.final.message).replace(/^-+\s*/, ""),
      sender: "bot",
      streaming: false,
    });
  }
  if (response.answer) {
    out.push({ text: response.answer, sender: "bot", streaming: false });
  }
  return out;
}

function parseAssistantContent(content) {
  try {
    const data = JSON.parse(content);
    if (data && typeof data === "object") {
      return responseToMessages(data);
    }
  } catch {
    /* plain text */
  }
  if (content?.trim()) {
    return [{ text: content, sender: "bot", streaming: false }];
  }
  return [];
}

function turnsToMessages(turns) {
  const messages = [];
  for (const turn of turns) {
    const idx = turns.indexOf(turn);
    if (turn.role === "user") {
      messages.push({ text: turn.content, sender: "user", turnIndex: idx });
    } else if (turn.role === "assistant") {
      const parsed = parseAssistantContent(turn.content).map((m) => ({ ...m, turnIndex: idx }));
      messages.push(...parsed);
    }
  }
  return messages;
}

/* ── Product Detail Panel ────────────────────── */
function ProductPanel({ product, onClose }) {
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onClose]);

  if (!product) return null;

  return (
    <>
      <div className="panel-overlay" onClick={onClose} />
      <aside className="product-panel">
        <button className="panel-close" onClick={onClose} aria-label="Close panel">
          <X size={16} />
        </button>
        {product.image && (
          <div className="panel-image">
            <img src={product.image} alt={product.title} />
          </div>
        )}
        <div className="panel-body">
          <h2 className="panel-title">{product.title}</h2>
          {product.price != null && <div className="panel-price">Price: {product.price}</div>}
          {product.description && <p className="panel-description">{product.description}</p>}
          {product.url && (
            <a href={product.url} target="_blank" rel="noopener noreferrer" className="panel-cta">
              View product <ArrowUpRight size={15} />
            </a>
          )}
        </div>
      </aside>
    </>
  );
}

function ProductCard({ product, onClick }) {
  return (
    <div className="product-card" onClick={() => onClick(product)}>
      {product.image && (
        <div className="product-card-image">
          <img src={product.image} alt={product.title} />
        </div>
      )}
      <div className="product-card-body">
        <div className="product-card-title">{product.title}</div>
        {product.price != null && <div className="product-card-meta">Price: {product.price}</div>}
        {product.description && (
          <div className="product-card-description">{product.description}</div>
        )}
        <div className="product-card-footer">
          <span className="product-card-link">
            View details <ArrowUpRight size={13} />
          </span>
        </div>
      </div>
    </div>
  );
}

const SUGGESTIONS = [
  "Best noise-cancelling headphones under $200",
  "Lightweight laptops for students",
  "Top rated running shoes for beginners",
  "Compact mirrorless cameras for travel",
];

function ChatSidebar({
  chats,
  activeId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  collapsed,
  onToggle,
  mobile,
  onBack,
}) {
  return (
    <aside className={`sidebar ${collapsed ? "sidebar-collapsed" : ""}`}>
      <div className="sidebar-top">
        <button type="button" className="sidebar-brand" onClick={onNewChat}>
          <span className="brand-mark">P</span>
          {!collapsed && <span className="brand-text">ProductGPT</span>}
        </button>
        <button
          type="button"
          className="icon-btn"
          onClick={() => {
            if (mobile && typeof onBack === "function") return onBack();
            return onToggle();
          }}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {mobile ? <ArrowLeft size={18} /> : <PanelLeftClose size={18} />}
        </button>
      </div>

      <button type="button" className="new-chat-btn" onClick={onNewChat}>
        <Plus size={18} />
        {!collapsed && <span>New chat</span>}
      </button>

      {!collapsed && (
        <div className="sidebar-section-label">Chats</div>
      )}

      <nav className="chat-list" aria-label="Chat history">
        {chats.length === 0 && !collapsed && (
          <p className="sidebar-empty">No conversations yet</p>
        )}
        {chats.map((chat) => (
          <div
            key={chat.id}
            className={`chat-list-item ${chat.id === activeId ? "active" : ""}`}
          >
            <button
              type="button"
              className="chat-list-button"
              onClick={() => onSelectChat(chat.id)}
              title={chat.title}
            >
              <MessageSquare size={16} className="chat-list-icon" />
              {!collapsed && <span className="chat-list-title">{chat.title}</span>}
            </button>
            {!collapsed && (
              <button
                type="button"
                className="chat-delete-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteChat(chat.id);
                }}
                aria-label="Delete chat"
              >
                <Trash2 size={14} />
              </button>
            )}
          </div>
        ))}
      </nav>
    </aside>
  );
}

function App() {
  const [store, setStore] = useState(loadStore);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [progressMessage, setProgressMessage] = useState("");
  const [streamingIndex, setStreamingIndex] = useState(-1);
  const [streamingText, setStreamingText] = useState("");
  const [messageQueue, setMessageQueue] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [imageUploading, setImageUploading] = useState(false);
  // pendingImage holds { file, previewUrl } while user has picked an image but not yet submitted
  const [pendingImage, setPendingImage] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [mobileSidebar, setMobileSidebar] = useState(false);
  const imageInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  const backendUrl = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";
  const streamUrl = `${backendUrl}/api/chat/stream`;
  const historyUrl = `${backendUrl}/api/chat/history`;

  const activeChat = store.chats.find((c) => c.id === store.activeId);
  const sessionId = store.activeId;
  const showWelcome = messages.length === 0 && !loading;

  const persistChat = useCallback((chatId, nextMessages, title) => {
    setStore((prev) => {
      const chats = prev.chats.map((c) => {
        if (c.id !== chatId) return c;
        return {
          ...c,
          messages: nextMessages,
          title: title ?? c.title,
          updatedAt: Date.now(),
        };
      });
      const next = { ...prev, chats };
      saveStore(next);
      return next;
    });
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  useEffect(() => {
    if (messageQueue.length > 0 && streamingIndex === -1) {
      const nextMessage = messageQueue[0];
      setStreamingIndex(nextMessage.index);
      streamMessage(nextMessage.text, nextMessage.index);
      setMessageQueue((prev) => prev.slice(1));
    }
  }, [messageQueue, streamingIndex]);

  const streamMessage = (message, index) => {
    const words = message.split(" ");
    let currentWord = 0;
    const streamInterval = setInterval(() => {
      if (currentWord <= words.length) {
        setStreamingText(words.slice(0, currentWord).join(" "));
        currentWord++;
      } else {
        clearInterval(streamInterval);
        setStreamingIndex(-1);
        setStreamingText("");
        setMessages((prev) => {
          const updated = [...prev];
          if (updated[index]) updated[index].text = message;
          return updated;
        });
      }
    }, 20);
  };

  const fetchServerHistory = async (id) => {
    if (!historyUrl) return [];
    try {
      const res = await fetch(`${historyUrl}?session_id=${encodeURIComponent(id)}&limit=80`);
      if (!res.ok) return [];
      const data = await res.json();
      const msgs = turnsToMessages(data.messages || []);
      // fetch reactions and attach
      try {
        const rres = await fetch(`${backendUrl}/api/chat/reactions?session_id=${encodeURIComponent(id)}`);
        if (rres.ok) {
          const map = await rres.json();
          for (const m of msgs) {
            if (m.turnIndex != null) {
              const val = map[String(m.turnIndex)];
              if (val) m.reaction = val;
            }
          }
        }
      } catch {
        /* ignore */
      }
      return msgs;
    } catch {
      return [];
    }
  };

  const loadChat = useCallback(
    async (chatId) => {
      const chat = store.chats.find((c) => c.id === chatId);
      if (!chat) return;

      let nextMessages = chat.messages || [];
      if (nextMessages.length === 0) {
        nextMessages = await fetchServerHistory(chatId);
        if (nextMessages.length > 0) {
          persistChat(chatId, nextMessages, chat.title);
        }
      }

      setMessages(nextMessages);
      setStore((prev) => {
        const next = { ...prev, activeId: chatId };
        saveStore(next);
        return next;
      });
      setMobileSidebar(false);
    },
    [store.chats, persistChat]
  );

  useEffect(() => {
    if (store.activeId) {
      loadChat(store.activeId);
    } else {
      setMessages([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const theme = localStorage.getItem("pgpt_theme") || "light";
    if (theme === "dark") document.documentElement.classList.add("theme-dark");
    else document.documentElement.classList.remove("theme-dark");
  }, []);

  const handleNewChat = () => {
    const id = createId();
    const chat = {
      id,
      title: "New chat",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
    };
    setStore((prev) => {
      const next = {
        activeId: id,
        chats: [chat, ...prev.chats],
      };
      saveStore(next);
      return next;
    });
    setMessages([]);
    setInput("");
    setMobileSidebar(false);
  };

  const handleSelectChat = (chatId) => {
    if (chatId === store.activeId) {
      setMobileSidebar(false);
      return;
    }
    loadChat(chatId);
  };

  const handleDeleteChat = (chatId) => {
    setStore((prev) => {
      const chats = prev.chats.filter((c) => c.id !== chatId);
      const wasActive = prev.activeId === chatId;
      const nextActive = wasActive ? chats[0]?.id ?? null : prev.activeId;
      const next = { activeId: nextActive, chats };
      saveStore(next);

      if (wasActive) {
        if (nextActive) {
          const chat = chats.find((c) => c.id === nextActive);
          const local = chat?.messages || [];
          if (local.length) {
            setMessages(local);
          } else {
            fetchServerHistory(nextActive).then((msgs) => setMessages(msgs));
          }
        } else {
          setMessages([]);
        }
      }
      return next;
    });
  };

  const handleClosePanel = useCallback(() => setSelectedProduct(null), []);

  // Clear pending image and revoke object URL to avoid memory leaks
  const clearPendingImage = useCallback(() => {
    setPendingImage((prev) => {
      if (prev?.previewUrl) URL.revokeObjectURL(prev.previewUrl);
      return null;
    });
    if (imageInputRef.current) imageInputRef.current.value = null;
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    // Allow submit with image even if text is empty
    if ((!input.trim() && !pendingImage) || loading) return;

    // If there's a pending image, delegate to the image search handler
    if (pendingImage) {
      const file = pendingImage.file;
      clearPendingImage();
      await handleImageSearch(file);
      return;
    }

    let chatId = sessionId;
    if (!chatId) {
      chatId = createId();
      const chat = {
        id: chatId,
        title: titleFromMessage(input),
        createdAt: Date.now(),
        updatedAt: Date.now(),
        messages: [],
      };
      setStore((prev) => {
        const next = { activeId: chatId, chats: [chat, ...prev.chats] };
        saveStore(next);
        return next;
      });
    }

    const userText = input.trim();
    const userMessage = { text: userText, sender: "user" };
    const messagesWithUser = [...messages, userMessage];
    setMessages(messagesWithUser);
    persistChat(chatId, messagesWithUser, titleFromMessage(userText));
    setInput("");
    setLoading(true);
    setProgressMessage("");

    try {
      const res = await fetch(streamUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user: "user", message: userText, session_id: chatId }),
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finalMessages = messagesWithUser;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (raw === "[DONE]") break;
          if (raw.startsWith("[ERROR]")) {
            finalMessages = [
              ...finalMessages,
              { text: raw.slice(8).trim(), sender: "error" },
            ];
            setMessages(finalMessages);
            break;
          }

          let event;
          try {
            event = JSON.parse(raw);
          } catch {
            continue;
          }

          if (event.type === "progress") {
            setProgressMessage(event.message);
          }

          if (event.type === "error") {
            setProgressMessage("");
            finalMessages = [
              ...finalMessages,
              { text: event.message || "An error occurred.", sender: "error" },
            ];
            setMessages(finalMessages);
            break;
          }

          if (event.type === "result") {
            setProgressMessage("");
            const newMessages = responseToMessages(event.data).map((m) =>
              m.sender === "bot" ? { ...m, streaming: true } : m
            );
            finalMessages = [...finalMessages, ...newMessages];
            setMessages(finalMessages);

            const queuedMessages = newMessages
              .map((msg, idx) => ({
                text: msg.text,
                index: finalMessages.length - newMessages.length + idx,
                streaming: msg.streaming,
              }))
              .filter((msg) => msg.streaming && msg.text);
            setMessageQueue(queuedMessages);
          }
        }
      }

      persistChat(chatId, finalMessages);
    } catch (err) {
      console.error(err);
      const withError = [...messagesWithUser, { text: "An error occurred.", sender: "error" }];
      setMessages(withError);
      persistChat(chatId, withError);
    } finally {
      setLoading(false);
      setProgressMessage("");
    }
  };

  const handleImageSearch = async (file) => {
    if (!file) return;

    // Determine or create a chat session
    let chatId = sessionId;
    if (!chatId) {
      chatId = createId();
      const chat = { id: chatId, title: "Image search", createdAt: Date.now(), updatedAt: Date.now(), messages: [] };
      setStore((prev) => { const next = { activeId: chatId, chats: [chat, ...prev.chats] }; saveStore(next); return next; });
    }

    const queryText = input.trim();
    // Create a user message that includes caption text + an image indicator
    const imagePreviewUrl = URL.createObjectURL(file);
    const userMessage = {
      text: queryText || "Image search",
      sender: "user",
      imageSrc: imagePreviewUrl,
    };
    const messagesWithUser = [...messages, userMessage];
    setMessages(messagesWithUser);
    persistChat(chatId, messagesWithUser, titleFromMessage(queryText || "Image search"));
    setInput("");
    setImageUploading(true);

    try {
      const fd = new FormData();
      fd.append("image", file);
      fd.append("query", queryText || "");
      fd.append("top_k", "5");
      const res = await fetch(`${backendUrl}/api/vector/multimodal/search`, {
        method: "POST",
        body: fd,
      });
      if (!res.ok) throw new Error(`Image search failed: ${res.status}`);
      const data = await res.json();
      const items = (data.results || []).map((r) => normalizeProduct(r.document || r));
      const prodMsg = { sender: "products", items };
      const next = [...messagesWithUser, prodMsg];
      setMessages(next);
      persistChat(chatId, next);
    } catch (err) {
      console.error(err);
      const withError = [...messagesWithUser, { text: "Image search failed. Please try again.", sender: "error" }];
      setMessages(withError);
      persistChat(chatId, withError);
    } finally {
      setImageUploading(false);
    }
  };

  return (
    <div className="App layout">
      {mobileSidebar && <div className="sidebar-backdrop" onClick={() => setMobileSidebar(false)} />}

      <div className={`sidebar-wrap ${mobileSidebar ? "mobile-open" : ""}`}>
        <ChatSidebar
          chats={[...store.chats].sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0))}
          activeId={store.activeId}
          onNewChat={handleNewChat}
          onSelectChat={handleSelectChat}
          onDeleteChat={handleDeleteChat}
          collapsed={!sidebarOpen}
          onToggle={() => setSidebarOpen((v) => !v)}
          mobile={mobileSidebar}
          onBack={() => setMobileSidebar(false)}
        />
      </div>

      <main className="main-panel">
        <header className="top-bar">
          <button
            type="button"
            className="icon-btn mobile-only"
            onClick={() => setMobileSidebar(true)}
            aria-label="Open menu"
          >
            <Menu size={20} />
          </button>
          <h1 className="top-bar-title">{activeChat?.title || "ProductGPT"}</h1>
          <div className="top-bar-actions">
            <button
              type="button"
              className="icon-btn grouped"
              onClick={async () => {
                const current = await fetch(`${backendUrl}/api/model`).then((r) => r.json()).catch(() => ({}));
                const newModel = window.prompt("Set model name:", current.model || "");
                if (!newModel) return;
                try {
                  const res = await fetch(`${backendUrl}/api/model`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ model: newModel }),
                  });
                  if (res.ok) {
                    alert(`Model set to ${newModel}`);
                  } else {
                    alert("Failed to set model");
                  }
                } catch (err) {
                  console.error(err);
                }
              }}
              title="Switch model"
              aria-label="Model"
            >
              Model
            </button>
            <button
              type="button"
              className="icon-btn grouped"
              onClick={() => {
                const cur = localStorage.getItem("pgpt_theme") || "light";
                const next = cur === "light" ? "dark" : "light";
                localStorage.setItem("pgpt_theme", next);
                document.documentElement.classList.toggle("theme-dark", next === "dark");
              }}
              title="Toggle theme"
            >
              Theme
            </button>
            <button
              type="button"
              className="icon-btn grouped"
              onClick={async () => {
                if (!sessionId) return;
                try {
                  const res = await fetch(`${backendUrl}/api/chat/regenerate`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ session_id: sessionId, index: -1 }),
                  });
                  const data = await res.json();
                  if (data.results) {
                    // Merge result chunks into messages
                    const merged = [];
                    for (const chunk of data.results) {
                      try {
                        const ev = JSON.parse(chunk);
                        if (ev.type === "result") {
                          merged.push(...responseToMessages(ev.data));
                        }
                      } catch {}
                    }
                    const next = [...messages, ...merged];
                    setMessages(next);
                    persistChat(sessionId, next);
                  }
                } catch (err) {
                  console.error(err);
                }
              }}
              title="Regenerate last user message"
              aria-label="Regenerate"
              disabled={!sessionId || loading}
            >
              Regenerate
            </button>

            <button
              type="button"
              className="icon-btn grouped"
              onClick={async () => {
                if (!sessionId) return;
                const newText = window.prompt("Edit last user message:");
                if (!newText) return;
                try {
                  const res = await fetch(`${backendUrl}/api/chat/edit`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ session_id: sessionId, index: -1, new_message: newText }),
                  });
                  const data = await res.json();
                  if (data.results) {
                    const merged = [];
                    for (const chunk of data.results) {
                      try {
                        const ev = JSON.parse(chunk);
                        if (ev.type === "result") {
                          merged.push(...responseToMessages(ev.data));
                        }
                      } catch {}
                    }
                    const next = [...messages, ...merged];
                    setMessages(next);
                    persistChat(sessionId, next);
                  }
                } catch (err) {
                  console.error(err);
                }
              }}
              title="Edit last user message and regenerate"
              aria-label="Edit"
              disabled={!sessionId || loading}
            >
              Edit
            </button>

            <button
              type="button"
              className="icon-btn grouped"
              onClick={async () => {
                if (!sessionId) return;
                if (!confirm("Delete last user turn?")) return;
                try {
                  const res = await fetch(`${backendUrl}/api/chat/delete`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ session_id: sessionId, index: -1 }),
                  });
                  if (res.ok) {
                    // refresh history from server
                    const msgs = await fetch(`${historyUrl}?session_id=${encodeURIComponent(sessionId)}&limit=80`).then((r) => r.json()).catch(() => ({ messages: [] }));
                    const next = turnsToMessages(msgs.messages || []);
                    setMessages(next);
                    persistChat(sessionId, next);
                  } else {
                    alert("Failed to delete turn");
                  }
                } catch (err) {
                  console.error(err);
                }
              }}
              title="Delete last user turn"
              aria-label="Delete last"
              disabled={!sessionId || loading}
            >
              Delete
            </button>

            <button
              type="button"
              className="icon-btn grouped"
              onClick={() => {
                if (!sessionId) return;
                const payload = { id: sessionId, title: activeChat?.title, messages };
                const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `${activeChat?.title || sessionId}.json`;
                a.click();
                URL.revokeObjectURL(url);
              }}
              title="Export conversation JSON"
              aria-label="Export"
              disabled={!sessionId}
            >
              Export
            </button>
          </div>
        </header>

        <div className="chat-container">
          {showWelcome && (
            <div className="welcome">
              <h2>How can I help you shop today?</h2>
              <p className="welcome-sub">Ask about products, compare options, or get recommendations.</p>
              <div className="suggestions">
                {SUGGESTIONS.map((text, i) => (
                  <button
                    key={i}
                    type="button"
                    className="suggestion-chip"
                    onClick={() => setInput(text)}
                  >
                    {text}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="messages">
            {messages.map((message, index) =>
              message.sender === "products" ? (
                <div key={index} className="products-container">
                  {message.items.map((product, i) => (
                    <ProductCard key={i} product={product} onClick={setSelectedProduct} />
                  ))}
                </div>
              ) : (
                <div key={index} className={`message ${message.sender}`}>
                  {/* Image thumbnail for image-search user messages */}
                  {message.sender === "user" && message.imageSrc && (
                    <div className="message-image-preview">
                      <img src={message.imageSrc} alt="Uploaded" />
                    </div>
                  )}
                  {index === streamingIndex && message.streaming ? (
                    streamingText
                  ) : (
                    // Render code fences as pre blocks
                    (typeof message.text === "string" && message.text.match(/```[\s\S]*?```/)) ? (
                      <div className="code-block">
                        {message.text.split(/```(?:\w+)?/).map((part, i) => (
                          <pre key={i} className="code-snippet">{part}</pre>
                        ))}
                      </div>
                    ) : (
                      message.text
                    )
                  )}

                  {message.sender === "user" && message.turnIndex != null && (
                    <div className="message-actions user-actions">
                      <button
                        type="button"
                        onClick={async () => {
                          // regenerate this user turn
                          try {
                            const res = await fetch(`${backendUrl}/api/chat/regenerate`, {
                              method: "POST",
                              headers: { "Content-Type": "application/json" },
                              body: JSON.stringify({ session_id: sessionId, index: message.turnIndex }),
                            });
                            const data = await res.json();
                            if (data.results) {
                              const merged = [];
                              for (const chunk of data.results) {
                                try {
                                  const ev = JSON.parse(chunk);
                                  if (ev.type === "result") merged.push(...responseToMessages(ev.data));
                                } catch {}
                              }
                              const next = [...messages, ...merged];
                              setMessages(next);
                              persistChat(sessionId, next);
                            }
                          } catch (err) {
                            console.error(err);
                          }
                        }}
                        title="Regenerate this message"
                      >
                        Regenerate
                      </button>

                      <button
                        type="button"
                        onClick={async () => {
                          const newText = window.prompt("Edit message:", message.text || "");
                          if (!newText) return;
                          try {
                            const res = await fetch(`${backendUrl}/api/chat/edit`, {
                              method: "POST",
                              headers: { "Content-Type": "application/json" },
                              body: JSON.stringify({ session_id: sessionId, index: message.turnIndex, new_message: newText }),
                            });
                            const data = await res.json();
                            if (data.results) {
                              const merged = [];
                              for (const chunk of data.results) {
                                try {
                                  const ev = JSON.parse(chunk);
                                  if (ev.type === "result") merged.push(...responseToMessages(ev.data));
                                } catch {}
                              }
                              const next = [...messages, ...merged];
                              setMessages(next);
                              persistChat(sessionId, next);
                            }
                          } catch (err) {
                            console.error(err);
                          }
                        }}
                        title="Edit this message and regenerate"
                      >
                        Edit
                      </button>

                      <button
                        type="button"
                        onClick={async () => {
                          if (!confirm("Delete this message?")) return;
                          try {
                            const res = await fetch(`${backendUrl}/api/chat/delete`, {
                              method: "POST",
                              headers: { "Content-Type": "application/json" },
                              body: JSON.stringify({ session_id: sessionId, index: message.turnIndex }),
                            });
                            if (res.ok) {
                              const msgs = await fetch(`${historyUrl}?session_id=${encodeURIComponent(sessionId)}&limit=80`).then((r) => r.json()).catch(() => ({ messages: [] }));
                              const next = turnsToMessages(msgs.messages || []);
                              setMessages(next);
                              persistChat(sessionId, next);
                            } else {
                              alert("Delete failed");
                            }
                          } catch (err) {
                            console.error(err);
                          }
                        }}
                        title="Delete this message"
                      >
                        Delete
                      </button>
                    </div>
                  )}

                  {message.sender === "bot" && (
                    <div className="message-actions">
                      <button
                        type="button"
                        className={`reaction-btn ${message.reaction === "up" ? "active" : ""}`}
                        onClick={async () => {
                          const next = [...messages];
                          const newReaction = next[index].reaction === "up" ? null : "up";
                          next[index] = { ...next[index], reaction: newReaction };
                          setMessages(next);
                          if (sessionId) {
                            try {
                              await fetch(`${backendUrl}/api/chat/reaction`, {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ session_id: sessionId, index: message.turnIndex, reaction: newReaction }),
                              });
                            } catch (err) {
                              console.error(err);
                            }
                            persistChat(sessionId, next);
                          }
                        }}
                        title="Like"
                      >
                        👍
                      </button>
                      <button
                        type="button"
                        className={`reaction-btn ${message.reaction === "down" ? "active" : ""}`}
                        onClick={async () => {
                          const next = [...messages];
                          const newReaction = next[index].reaction === "down" ? null : "down";
                          next[index] = { ...next[index], reaction: newReaction };
                          setMessages(next);
                          if (sessionId) {
                            try {
                              await fetch(`${backendUrl}/api/chat/reaction`, {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ session_id: sessionId, index: message.turnIndex, reaction: newReaction }),
                              });
                            } catch (err) {
                              console.error(err);
                            }
                            persistChat(sessionId, next);
                          }
                        }}
                        title="Dislike"
                      >
                        👎
                      </button>
                    </div>
                  )}
                </div>
              )
            )}

            {loading && (
              <div className="progress-bubble">
                <div className="typing-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                {progressMessage && <span className="progress-label">{progressMessage}</span>}
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSubmit} className="input-form">
            {/* ── Pending image preview strip ── */}
            {pendingImage && (
              <div className="image-preview-strip">
                <div className="image-preview-thumb">
                  <img src={pendingImage.previewUrl} alt="Preview" />
                  <button
                    type="button"
                    className="image-preview-remove"
                    onClick={clearPendingImage}
                    aria-label="Remove image"
                  >
                    <X size={12} />
                  </button>
                </div>
                <span className="image-preview-name">{pendingImage.file.name}</span>
              </div>
            )}

            <div className="input-wrapper">
              {/* Hidden file input */}
              <input
                ref={imageInputRef}
                type="file"
                accept="image/*"
                style={{ display: "none" }}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (!f) return;
                  // Revoke any existing preview URL
                  if (pendingImage?.previewUrl) URL.revokeObjectURL(pendingImage.previewUrl);
                  setPendingImage({ file: f, previewUrl: URL.createObjectURL(f) });
                }}
              />
              {/* Image upload button */}
              <button
                type="button"
                className="image-upload-btn"
                title="Attach an image"
                onClick={() => imageInputRef.current?.click()}
                disabled={loading || imageUploading}
                aria-label="Upload image"
              >
                {imageUploading ? <Loader2 size={18} className="spin-icon" /> : <ImageIcon size={18} />}
              </button>

              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={pendingImage ? "Add a caption (optional)…" : "Message ProductGPT…"}
                disabled={loading || imageUploading}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
              />
              <button
                type="submit"
                disabled={loading || imageUploading || (!input.trim() && !pendingImage)}
                aria-label="Send message"
                className="send-button"
              >
                {loading ? <Loader2 size={16} className="spin-icon" /> : <Send size={16} />}
              </button>
            </div>
            <p className="input-hint">ProductGPT · Context is scoped to each conversation</p>
          </form>
        </div>
      </main>

      <ProductPanel product={selectedProduct} onClose={handleClosePanel} />
    </div>
  );
}

export default App;
