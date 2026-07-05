"""
TAR UMT AI Chatbot — GUI Version (ChatGPT-style)
==================================================
A sleek, ChatGPT-inspired Tkinter GUI for the TAR UMT chatbot.
Loads the saved TF-IDF vectorizer + DNN model and provides
an interactive chat experience focused on TAR UMT facility information.

Pre-trained assets required (from running DNN_model_train.py):
  - tarumt_chatbot_model.keras  (DNN model)
  - tokenizer.pickle            (TF-IDF vectorizer)
  - label_encoder.pickle        (Label encoder)
  - intents.json                (Responses)

Usage:
  python tarumtAiBot.py
"""

import json
import numpy as np
import pickle
import random
import os
import string
import difflib
import tkinter as tk
from tkinter import font as tkfont
import threading
from datetime import datetime


# ============================================================
#  LOAD & PREDICT — Uses saved TF-IDF + DNN model
# ============================================================
REQUIRED_FILES = [
    "tarumt_chatbot_model.keras",
    "tokenizer.pickle",
    "label_encoder.pickle",
    "intents.json",
]
CONFIDENCE_THRESHOLD = 0.55


def load_assets():
    """
    Load the saved TF-IDF vectorizer + DNN model + label encoder + responses.
    Returns (model, vectorizer, encoder, responses).
    """
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(f"  • {f}" for f in missing)
            + "\n\nPlease run 'DNN_model_train.py' first to train the model."
        )

    from tensorflow.keras.models import load_model as keras_load
    model = keras_load("tarumt_chatbot_model.keras")

    with open("tokenizer.pickle", "rb") as f:
        vectorizer = pickle.load(f)
    with open("label_encoder.pickle", "rb") as f:
        encoder = pickle.load(f)
    with open("intents.json", encoding="utf-8") as f:
        data = json.load(f)

    responses = {intent["tag"]: intent["responses"] for intent in data["intents"]}
    return model, vectorizer, encoder, responses


def predict_intent(text, model, vectorizer, encoder, responses,
                   threshold=CONFIDENCE_THRESHOLD):
    """
    Predict intent using the TF-IDF vectorizer + DNN model.
    Returns (response, confidence, tag).
    """
    cleaned = text.lower().strip()
    cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))
    
    # Auto-correct misspelled words using ONLY single-word (unigram) vocabulary
    # We filter out bigrams/trigrams (which contain spaces) to avoid false matches
    unigram_vocab = [w for w in vectorizer.vocabulary_.keys() if ' ' not in w]
    corrected_words = []
    for word in cleaned.split():
        if word in vectorizer.vocabulary_:
            corrected_words.append(word)
        else:
            # Find closest matching unigram (e.g., 'libray' -> 'library')
            matches = difflib.get_close_matches(word, unigram_vocab, n=1, cutoff=0.6)
            if matches:
                corrected_words.append(matches[0])
            else:
                corrected_words.append(word)
                
    cleaned = ' '.join(corrected_words)
    
    input_vector = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(input_vector, verbose=0)
    confidence = float(np.max(prediction))
    tag = encoder.inverse_transform([np.argmax(prediction)])[0]

    if confidence > threshold:
        response = random.choice(responses[tag])
    else:
        response = (
            "I'm not sure about that. Could you rephrase your question? "
            "Try asking about TAR UMT facilities like the Library, CITC, Sports Complex, Clubhouse, or Arena."
        )

    return response, confidence, tag


# ============================================================
#  COLOUR PALETTE — ChatGPT light-mode inspired
# ============================================================
SIDEBAR_BG     = "#171717"       # sidebar stays dark (like ChatGPT)
SIDEBAR_HOVER  = "#2f2f2f"
MAIN_BG        = "#ffffff"       # white main background
INPUT_BG       = "#f4f4f4"       # light grey input pill
INPUT_BORDER   = "#d9d9d9"       # subtle border
SEND_BG        = "#d9d9d9"       # send button default
SEND_ACTIVE    = "#c0c0c0"       # send button hover
TEXT_PRIMARY    = "#0d0d0d"       # near-black text
TEXT_SECONDARY  = "#6e6e6e"      # secondary text
TEXT_MUTED      = "#999999"      # muted / placeholder text
TEXT_SIDEBAR    = "#ececec"       # sidebar text stays light
SIDEBAR_ICON    = "#b4b4b4"
DIVIDER         = "#e5e5e5"      # light divider for main area
DIVIDER_SIDEBAR = "#2f2f2f"      # dark divider for sidebar
USER_AVATAR_BG  = "#9b59b6"
BOT_AVATAR_BG   = "#19c37d"
USER_BUBBLE_BG  = "#f4f4f4"      # user message bubble
BOT_BUBBLE_BG   = "#ffffff"      # bot message (same as background)
SCROLLBAR_BG    = "#f4f4f4"
SCROLLBAR_FG    = "#c0c0c0"
HOVER_BG        = "#f4f4f4"      # chip hover

# Fonts
FONT_FAMILY    = "Segoe UI"



# ============================================================
#  AUTO-HIDING SCROLLBAR
# ============================================================
class AutoScrollbar(tk.Scrollbar):
    """A scrollbar that hides itself when not needed."""
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.pack_forget()
        else:
            if self.cget("orient") == tk.VERTICAL:
                self.pack(side=tk.RIGHT, fill=tk.Y)
            else:
                self.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Scrollbar.set(self, lo, hi)


# ============================================================
#  MAIN GUI CLASS
# ============================================================
class ChatGPTStyleGUI:
    """ChatGPT-inspired Tkinter GUI for TAR UMT AI Chatbot."""

    CONFIDENCE_THRESHOLD = 0.55

    def __init__(self, root):
        self.root = root
        self.root.title("TAR UMT AI Chatbot")
        self.root.geometry("1100x720")
        self.root.minsize(800, 550)
        self.root.configure(bg=MAIN_BG)

        # State
        self.model = None
        self.vectorizer = None
        self.encoder = None
        self.responses = None
        self.is_loading = True
        self.messages = []           # list of (role, text) tuples
        self.chat_sessions = []      # sidebar history
        self.current_session = 0
        self.sidebar_visible = True

        # Fonts (created after root exists)
        self.font_heading  = (FONT_FAMILY, 22, "bold")
        self.font_body     = (FONT_FAMILY, 14)
        self.font_small    = (FONT_FAMILY, 12)
        self.font_tiny     = (FONT_FAMILY, 10)
        self.font_input    = (FONT_FAMILY, 14)
        self.font_sidebar  = (FONT_FAMILY, 12)
        self.font_btn      = (FONT_FAMILY, 13)

        self._build_ui()
        self._load_model_async()

    # ────────────────────────────────────────────
    #  BUILD THE FULL UI
    # ────────────────────────────────────────────
    def _build_ui(self):
        # Top-level horizontal split: sidebar | main
        self.container = tk.Frame(self.root, bg=MAIN_BG)
        self.container.pack(fill=tk.BOTH, expand=True)

        self._build_sidebar()
        self._build_main_area()

    # ── SIDEBAR ──────────────────────────────────
    def _build_sidebar(self):
        self.sidebar = tk.Frame(self.container, bg=SIDEBAR_BG, width=260)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        # ── Top section: New Chat button ──
        top = tk.Frame(self.sidebar, bg=SIDEBAR_BG)
        top.pack(fill=tk.X, padx=12, pady=(14, 6))

        # Sidebar toggle (hamburger)
        self.toggle_btn = tk.Label(
            top, text="☰", fg=TEXT_SIDEBAR, bg=SIDEBAR_BG,
            font=(FONT_FAMILY, 18), cursor="hand2"
        )
        self.toggle_btn.pack(side=tk.LEFT)
        self.toggle_btn.bind("<Button-1>", lambda e: self._toggle_sidebar())

        # New Chat button
        new_chat_btn = tk.Label(
            top, text="✎", fg=TEXT_SIDEBAR, bg=SIDEBAR_BG,
            font=(FONT_FAMILY, 18), cursor="hand2"
        )
        new_chat_btn.pack(side=tk.RIGHT)
        new_chat_btn.bind("<Button-1>", lambda e: self._new_chat())

        # ── Separator ──
        tk.Frame(self.sidebar, bg=DIVIDER_SIDEBAR, height=1).pack(fill=tk.X, padx=12, pady=8)

        # ── Chat history list ──
        history_label = tk.Label(
            self.sidebar, text="Your chats", fg=TEXT_MUTED, bg=SIDEBAR_BG,
            font=(FONT_FAMILY, 10), anchor="w"
        )
        history_label.pack(fill=tk.X, padx=16, pady=(4, 2))

        self.history_frame = tk.Frame(self.sidebar, bg=SIDEBAR_BG)
        self.history_frame.pack(fill=tk.BOTH, expand=True, padx=8)

        # Welcome item
        self._add_history_item("TAR UMT Chatbot Session", active=True)

        # ── Bottom: branding ──
        bottom = tk.Frame(self.sidebar, bg=SIDEBAR_BG)
        bottom.pack(fill=tk.X, side=tk.BOTTOM, padx=12, pady=12)

        tk.Frame(bottom, bg=DIVIDER_SIDEBAR, height=1).pack(fill=tk.X, pady=(0, 10))

        brand = tk.Label(
            bottom, text="TAR UMT AI Chatbot v2.0",
            fg=TEXT_MUTED, bg=SIDEBAR_BG, font=(FONT_FAMILY, 9)
        )
        brand.pack(anchor="w")
        powered = tk.Label(
            bottom, text="Powered by TensorFlow + TF-IDF",
            fg=TEXT_MUTED, bg=SIDEBAR_BG, font=(FONT_FAMILY, 9)
        )
        powered.pack(anchor="w")

    def _add_history_item(self, title, active=False):
        """Add a chat session to the sidebar history."""
        bg = SIDEBAR_HOVER if active else SIDEBAR_BG
        item = tk.Frame(self.history_frame, bg=bg, cursor="hand2")
        item.pack(fill=tk.X, pady=1)

        lbl = tk.Label(
            item, text=f"  💬  {title}", fg=TEXT_SIDEBAR, bg=bg,
            font=self.font_sidebar, anchor="w", padx=8, pady=8
        )
        lbl.pack(fill=tk.X)

        # Hover effects
        for widget in (item, lbl):
            widget.bind("<Enter>", lambda e, f=item, l=lbl: (
                f.configure(bg=SIDEBAR_HOVER), l.configure(bg=SIDEBAR_HOVER)))
            widget.bind("<Leave>", lambda e, f=item, l=lbl, a=active: (
                f.configure(bg=SIDEBAR_HOVER if a else SIDEBAR_BG),
                l.configure(bg=SIDEBAR_HOVER if a else SIDEBAR_BG)))

    def _toggle_sidebar(self):
        """Show/hide the sidebar."""
        if self.sidebar_visible:
            self.sidebar.pack_forget()
            self.sidebar_visible = False
        else:
            self.sidebar.pack(side=tk.LEFT, fill=tk.Y, before=self.main_area)
            self.sidebar_visible = True

    # ── MAIN AREA ────────────────────────────────
    def _build_main_area(self):
        self.main_area = tk.Frame(self.container, bg=MAIN_BG)
        self.main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── Top bar (model name + status) ──
        topbar = tk.Frame(self.main_area, bg=MAIN_BG, height=50)
        topbar.pack(fill=tk.X)
        topbar.pack_propagate(False)

        # Sidebar toggle for when sidebar is hidden
        self.main_toggle = tk.Label(
            topbar, text="☰", fg=TEXT_SECONDARY, bg=MAIN_BG,
            font=(FONT_FAMILY, 16), cursor="hand2", padx=16
        )
        self.main_toggle.pack(side=tk.LEFT, pady=10)
        self.main_toggle.bind("<Button-1>", lambda e: self._toggle_sidebar())

        title_label = tk.Label(
            topbar, text="TAR UMT AI Chatbot", fg=TEXT_PRIMARY, bg=MAIN_BG,
            font=(FONT_FAMILY, 14, "bold")
        )
        title_label.pack(side=tk.LEFT, pady=10)

        self.status_dot = tk.Label(
            topbar, text="", fg=TEXT_MUTED, bg=MAIN_BG,
            font=(FONT_FAMILY, 10)
        )
        self.status_dot.pack(side=tk.LEFT, padx=(8, 0), pady=10)

        # ── Chat scroll area ──
        self.chat_outer = tk.Frame(self.main_area, bg=MAIN_BG)
        self.chat_outer.pack(fill=tk.BOTH, expand=True)

        # Canvas + scrollbar for scrollable chat
        self.chat_canvas = tk.Canvas(
            self.chat_outer, bg=MAIN_BG, highlightthickness=0, bd=0
        )
        self.chat_scrollbar = AutoScrollbar(
            self.chat_outer, orient=tk.VERTICAL, command=self.chat_canvas.yview,
            bg=SCROLLBAR_BG, troughcolor=MAIN_BG, activebackground=SCROLLBAR_FG,
            width=8, relief=tk.FLAT
        )
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)
        self.chat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Inner frame that holds all messages
        self.chat_inner = tk.Frame(self.chat_canvas, bg=MAIN_BG)
        self.chat_window = self.chat_canvas.create_window(
            (0, 0), window=self.chat_inner, anchor="n"
        )

        # Make the inner frame resize with canvas
        self.chat_canvas.bind("<Configure>", self._on_canvas_configure)
        self.chat_inner.bind("<Configure>",
                             lambda e: self.chat_canvas.configure(
                                 scrollregion=self.chat_canvas.bbox("all")))

        # Mouse wheel scrolling
        self.chat_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # ── Welcome screen (shown initially) ──
        self.welcome_frame = tk.Frame(self.chat_inner, bg=MAIN_BG)
        self.welcome_frame.pack(expand=True, fill=tk.BOTH)

        spacer = tk.Frame(self.welcome_frame, bg=MAIN_BG, height=140)
        spacer.pack()

        self.welcome_label = tk.Label(
            self.welcome_frame,
            text="What can I help you with?",
            fg=TEXT_PRIMARY, bg=MAIN_BG,
            font=self.font_heading
        )
        self.welcome_label.pack(pady=(0, 8))

        self.welcome_sub = tk.Label(
            self.welcome_frame,
            text="Ask me about TAR UMT facilities — Library, CITC, Sports Complex, Clubhouse, Arena & more",
            fg=TEXT_MUTED, bg=MAIN_BG,
            font=self.font_small
        )
        self.welcome_sub.pack(pady=(0, 30))

        # Quick-action suggestion chips
        chips_frame = tk.Frame(self.welcome_frame, bg=MAIN_BG)
        chips_frame.pack()

        suggestions = [
            "📚  Library info",
            "🖥️  CITC computer labs",
            "🏟️  Sports Complex",
            "🏢  Arena facilities",
        ]
        # Two rows of chips
        row1 = tk.Frame(chips_frame, bg=MAIN_BG)
        row1.pack(pady=4)
        row2 = tk.Frame(chips_frame, bg=MAIN_BG)
        row2.pack(pady=4)

        for i, text in enumerate(suggestions):
            parent = row1 if i < 2 else row2
            chip = tk.Label(
                parent, text=text, fg=TEXT_SECONDARY, bg=MAIN_BG,
                font=self.font_small, padx=20, pady=10, cursor="hand2",
                relief=tk.FLAT, bd=1,
                highlightbackground=INPUT_BORDER, highlightthickness=1
            )
            chip.pack(side=tk.LEFT, padx=6)
            chip.bind("<Enter>", lambda e, c=chip: c.configure(bg=HOVER_BG))
            chip.bind("<Leave>", lambda e, c=chip: c.configure(bg=MAIN_BG))
            chip.bind("<Button-1>", lambda e, t=text: self._use_suggestion(t))

        # ── Messages container (hidden initially, shown after first message) ──
        self.messages_frame = tk.Frame(self.chat_inner, bg=MAIN_BG)

        # ── Input area at the bottom ──
        self._build_input_area()

    def _build_input_area(self):
        """Build the ChatGPT-style pill input bar."""
        # Outer container pinned to bottom
        input_outer = tk.Frame(self.main_area, bg=MAIN_BG)
        input_outer.pack(fill=tk.X, side=tk.BOTTOM, padx=0, pady=(0, 0))

        # Footer text
        footer = tk.Label(
            input_outer,
            text="TAR UMT AI Chatbot may produce inaccurate information. Consider verifying important details.",
            fg=TEXT_MUTED, bg=MAIN_BG, font=(FONT_FAMILY, 9)
        )
        footer.pack(pady=(0, 8))

        # Pill-shaped input container
        # We simulate a rounded rectangle with a Frame
        input_wrapper = tk.Frame(input_outer, bg=MAIN_BG)
        input_wrapper.pack(fill=tk.X, padx=80, pady=(0, 16))

        # The pill container
        self.pill = tk.Frame(
            input_wrapper, bg=INPUT_BG,
            highlightbackground=INPUT_BORDER, highlightthickness=1,
            bd=0, padx=4, pady=4
        )
        self.pill.pack(fill=tk.X)

        # Inner layout of the pill
        pill_inner = tk.Frame(self.pill, bg=INPUT_BG)
        pill_inner.pack(fill=tk.X, padx=8, pady=6)

        # Plus button (left side)
        plus_btn = tk.Label(
            pill_inner, text="＋", fg=TEXT_SECONDARY, bg=INPUT_BG,
            font=(FONT_FAMILY, 16), cursor="hand2", padx=4
        )
        plus_btn.pack(side=tk.LEFT)

        # Text entry
        self.user_input = tk.Entry(
            pill_inner, bg=INPUT_BG, fg=TEXT_PRIMARY,
            font=self.font_input, relief=tk.FLAT,
            insertbackground=TEXT_PRIMARY,
            selectbackground="#b3d9ff", selectforeground=TEXT_PRIMARY
        )
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
        self.user_input.insert(0, "Ask anything")
        self.user_input.configure(fg=TEXT_MUTED)

        # Placeholder behaviour
        self.user_input.bind("<FocusIn>", self._on_focus_in)
        self.user_input.bind("<FocusOut>", self._on_focus_out)
        self.user_input.bind("<Return>", self._on_send)
        self.user_input.bind("<KeyRelease>", self._update_send_button)

        # Send button (right side) — circular
        self.send_btn = tk.Label(
            pill_inner, text="↑", fg="#ffffff", bg=SEND_BG,
            font=(FONT_FAMILY, 14, "bold"), width=2, height=1,
            cursor="hand2", relief=tk.FLAT
        )
        self.send_btn.pack(side=tk.RIGHT, padx=(4, 0))
        self.send_btn.bind("<Button-1>", self._on_send)
        self.send_btn.bind("<Enter>",
                           lambda e: self.send_btn.configure(bg=SEND_ACTIVE))
        self.send_btn.bind("<Leave>",
                           lambda e: self.send_btn.configure(bg=SEND_BG))

    # ────────────────────────────────────────────
    #  INPUT PLACEHOLDER LOGIC
    # ────────────────────────────────────────────
    def _on_focus_in(self, event=None):
        if self.user_input.get() == "Ask anything":
            self.user_input.delete(0, tk.END)
            self.user_input.configure(fg=TEXT_PRIMARY)

    def _on_focus_out(self, event=None):
        if not self.user_input.get().strip():
            self.user_input.delete(0, tk.END)
            self.user_input.insert(0, "Ask anything")
            self.user_input.configure(fg=TEXT_MUTED)

    def _update_send_button(self, event=None):
        """Highlight send button when there's text."""
        text = self.user_input.get().strip()
        if text and text != "Ask anything":
            self.send_btn.configure(bg=TEXT_PRIMARY, fg="#ffffff")
        else:
            self.send_btn.configure(bg=SEND_BG, fg="#ffffff")

    # ────────────────────────────────────────────
    #  CANVAS / SCROLL HELPERS
    # ────────────────────────────────────────────
    def _on_canvas_configure(self, event):
        self.chat_canvas.itemconfig(self.chat_window, width=event.width)

    def _on_mousewheel(self, event):
        self.chat_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _scroll_to_bottom(self):
        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)

    # ────────────────────────────────────────────
    #  MODEL LOADING
    # ────────────────────────────────────────────
    def _load_model_async(self):
        def _load():
            try:
                self.model, self.vectorizer, self.encoder, self.responses = load_assets()
                self.is_loading = False
                self.root.after(0, self._on_model_loaded)
            except Exception as e:
                self.root.after(0, lambda: self._on_model_error(str(e)))
        threading.Thread(target=_load, daemon=True).start()

    def _on_model_loaded(self):
        num_intents = len(self.responses)
        self.status_dot.configure(text=f"● Online  •  {num_intents} intents", fg="#19c37d")
        self.user_input.focus_set()

    def _on_model_error(self, error_msg):
        self.status_dot.configure(text="● Offline — Model not loaded", fg="#ef4444")
        from tkinter import messagebox
        messagebox.showerror("Model Load Error", error_msg)

    # ────────────────────────────────────────────
    #  SUGGESTIONS
    # ────────────────────────────────────────────
    def _use_suggestion(self, text):
        # Strip the emoji prefix
        clean = text.split("  ", 1)[-1] if "  " in text else text
        self.user_input.delete(0, tk.END)
        self.user_input.configure(fg=TEXT_PRIMARY)
        self.user_input.insert(0, clean)
        self._on_send()

    # ────────────────────────────────────────────
    #  NEW CHAT
    # ────────────────────────────────────────────
    def _new_chat(self):
        self.messages.clear()
        # Clear messages frame
        for w in self.messages_frame.winfo_children():
            w.destroy()
        self.messages_frame.pack_forget()
        # Show welcome again
        self.welcome_frame.pack(expand=True, fill=tk.BOTH)
        self._scroll_to_bottom()

    # ────────────────────────────────────────────
    #  SENDING MESSAGES
    # ────────────────────────────────────────────
    def _on_send(self, event=None):
        if self.is_loading:
            return

        text = self.user_input.get().strip()
        if not text or text == "Ask anything":
            return

        self.user_input.delete(0, tk.END)
        self.user_input.configure(fg=TEXT_PRIMARY)

        # Hide welcome, show messages
        if self.welcome_frame.winfo_ismapped():
            self.welcome_frame.pack_forget()
            self.messages_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))

        # Exit commands
        if text.lower() in ["quit", "exit", "bye"]:
            self._display_user_message(text)
            self._display_bot_message("Goodbye! Have a great day at TAR UMT! 🎓")
            self.root.after(2000, self.root.destroy)
            return

        self._display_user_message(text)
        self.messages.append(("user", text))

        # Show typing indicator
        typing_frame = self._show_typing_indicator()

        # Predict in background
        threading.Thread(target=self._predict, args=(text, typing_frame), daemon=True).start()

    def _predict(self, text, typing_frame):
        response, confidence, tag = predict_intent(
            text, self.model, self.vectorizer, self.encoder, self.responses
        )
        self.root.after(0, lambda: self._on_prediction_done(response, confidence, tag, typing_frame))

    def _on_prediction_done(self, response, confidence, tag, typing_frame):
        # Remove typing indicator
        typing_frame.destroy()
        self._display_bot_message(response, confidence, tag)
        self.messages.append(("bot", response))

    # ────────────────────────────────────────────
    #  MESSAGE DISPLAY — ChatGPT style
    # ────────────────────────────────────────────
    def _display_user_message(self, text):
        """Display a user message row (right-aligned avatar + text)."""
        row = tk.Frame(self.messages_frame, bg=MAIN_BG)
        row.pack(fill=tk.X, padx=60, pady=(16, 4))

        # Content (right-aligned appearance)
        content = tk.Frame(row, bg=MAIN_BG)
        content.pack(anchor="e")

        # Avatar circle
        avatar_frame = tk.Frame(content, bg=MAIN_BG)
        avatar_frame.pack(side=tk.RIGHT, anchor="n", padx=(12, 0), pady=(2, 0))

        avatar = tk.Canvas(avatar_frame, width=30, height=30, bg=MAIN_BG,
                           highlightthickness=0)
        avatar.pack()
        avatar.create_oval(2, 2, 28, 28, fill=USER_AVATAR_BG, outline="")
        avatar.create_text(15, 15, text="Y", fill="white",
                           font=(FONT_FAMILY, 11, "bold"))

        # Message text with background
        msg_container = tk.Frame(content, bg=USER_BUBBLE_BG, padx=16, pady=10)
        msg_container.pack(side=tk.RIGHT)

        msg = tk.Label(
            msg_container, text=text, fg=TEXT_PRIMARY, bg=USER_BUBBLE_BG,
            font=self.font_body, wraplength=500, justify=tk.LEFT, anchor="w"
        )
        msg.pack()

        self._scroll_to_bottom()

    def _display_bot_message(self, text, confidence=None, tag=None):
        """Display a bot message row (left-aligned avatar + text)."""
        row = tk.Frame(self.messages_frame, bg=MAIN_BG)
        row.pack(fill=tk.X, padx=60, pady=(16, 4))

        content = tk.Frame(row, bg=MAIN_BG)
        content.pack(anchor="w")

        # Avatar circle
        avatar_frame = tk.Frame(content, bg=MAIN_BG)
        avatar_frame.pack(side=tk.LEFT, anchor="n", padx=(0, 12), pady=(2, 0))

        avatar = tk.Canvas(avatar_frame, width=30, height=30, bg=MAIN_BG,
                           highlightthickness=0)
        avatar.pack()
        avatar.create_oval(2, 2, 28, 28, fill=BOT_AVATAR_BG, outline="")
        avatar.create_text(15, 15, text="T", fill="white",
                           font=(FONT_FAMILY, 11, "bold"))

        # Message text
        text_frame = tk.Frame(content, bg=MAIN_BG)
        text_frame.pack(side=tk.LEFT)

        msg = tk.Label(
            text_frame, text=text, fg=TEXT_PRIMARY, bg=MAIN_BG,
            font=self.font_body, wraplength=520, justify=tk.LEFT, anchor="w"
        )
        msg.pack(anchor="w")

        # Confidence & intent tag (subtle, below the message)
        if confidence is not None:
            conf_pct = f"{confidence * 100:.0f}%"
            conf_color = "#19c37d" if confidence > self.CONFIDENCE_THRESHOLD else "#ef4444"
            meta = tk.Label(
                text_frame,
                text=f"Confidence: {conf_pct}  ·  Intent: {tag}",
                fg=TEXT_MUTED, bg=MAIN_BG, font=self.font_tiny, anchor="w"
            )
            meta.pack(anchor="w", pady=(4, 0))

        self._scroll_to_bottom()

    def _show_typing_indicator(self):
        """Show a 'thinking' animation row."""
        row = tk.Frame(self.messages_frame, bg=MAIN_BG)
        row.pack(fill=tk.X, padx=60, pady=(16, 4))

        content = tk.Frame(row, bg=MAIN_BG)
        content.pack(anchor="w")

        avatar_frame = tk.Frame(content, bg=MAIN_BG)
        avatar_frame.pack(side=tk.LEFT, anchor="n", padx=(0, 12), pady=(2, 0))

        avatar = tk.Canvas(avatar_frame, width=30, height=30, bg=MAIN_BG,
                           highlightthickness=0)
        avatar.pack()
        avatar.create_oval(2, 2, 28, 28, fill=BOT_AVATAR_BG, outline="")
        avatar.create_text(15, 15, text="T", fill="white",
                           font=(FONT_FAMILY, 11, "bold"))

        dots = tk.Label(
            content, text="●  ●  ●", fg=TEXT_MUTED, bg=MAIN_BG,
            font=(FONT_FAMILY, 14)
        )
        dots.pack(side=tk.LEFT, pady=4)

        self._scroll_to_bottom()

        # Animate the dots
        self._animate_dots(dots, 0)

        return row

    def _animate_dots(self, label, step):
        """Simple dot animation for the typing indicator."""
        if not label.winfo_exists():
            return
        patterns = ["●  ·  ·", "·  ●  ·", "·  ·  ●", "·  ●  ·"]
        label.configure(text=patterns[step % len(patterns)])
        self.root.after(350, lambda: self._animate_dots(label, step + 1))


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatGPTStyleGUI(root)
    root.mainloop()
