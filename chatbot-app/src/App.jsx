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
    if (turn.role === "user") {
      messages.push({ text: turn.content, sender: "user" });
    } else if (turn.role === "assistant") {
      messages.push(...parseAssistantContent(turn.content));
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
          onClick={onToggle}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <PanelLeftClose size={18} />
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
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [mobileSidebar, setMobileSidebar] = useState(false);
  const messagesEndRef = useRef(null);

  const backendUrl = import.meta.env.VITE_BACKEND_URL || "";
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
      return turnsToMessages(data.messages || []);
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

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

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
                  {index === streamingIndex && message.streaming ? streamingText : message.text}
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
            <div className="input-wrapper">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Message ProductGPT…"
                disabled={loading}
              />
              <button
                type="submit"
                disabled={loading || !input.trim()}
                aria-label="Send message"
                className="send-button"
              >
                <Send size={16} />
              </button>
            </div>
            <p className="input-hint">ProductGPT remembers context within each chat when Redis is enabled.</p>
          </form>
        </div>
      </main>

      <ProductPanel product={selectedProduct} onClose={handleClosePanel} />
    </div>
  );
}

export default App;
