/* ============================================================
   SpamSense AI  –  app.js
   Client-side Naive Bayes simulation for the live demo section.
   Mirrors the exact algorithm in SRC/predict.py so users can
   see a real-time result without running Python.
   ============================================================ */

// ── Training data (mirrors data/dataset.txt) ─────────────────
const DATASET = [
  ["spam","Win money now!!!"],
  ["spam","Congratulations you have won a lottery"],
  ["spam","Claim your free prize now"],
  ["spam","Get rich quick with this trick"],
  ["spam","Exclusive offer just for you"],
  ["spam","You have been selected for a cash reward"],
  ["spam","Earn money from home easily"],
  ["spam","Limited time offer act now"],
  ["spam","Free entry in 1000 dollar contest"],
  ["spam","Click here to claim your reward"],
  ["spam","Urgent response needed to claim prize"],
  ["spam","Get a loan approved instantly"],
  ["spam","Lowest price guaranteed click now"],
  ["spam","You are a lucky winner"],
  ["spam","Double your income in a week"],
  ["spam","Buy now and save big"],
  ["spam","You won a free gift card today"],
  ["spam","Act fast to claim your bonus cash"],
  ["spam","Huge discount apply now"],
  ["spam","Special promotion only for you"],
  ["spam","Free trial no credit card required"],
  ["spam","Make money fast guaranteed results"],
  ["spam","Call now to claim your reward"],
  ["spam","Congratulations your prize is ready"],
  ["spam","Investment opportunity returns guaranteed"],
  ["spam","You have been pre selected for a reward"],
  ["spam","Earn passive income from home"],
  ["spam","Risk free money back guarantee"],
  ["spam","Your account has been flagged click to verify"],
  ["spam","Dear winner please provide your details"],
  ["spam","Buy pills online at low cost"],
  ["spam","You qualified for our special credit offer"],
  ["spam","Instant cash advance approved"],
  ["spam","Million dollar opportunity waiting for you"],
  ["spam","Order now and get free shipping"],
  ["spam","Click the link to claim your refund"],
  ["spam","Hot singles in your area tonight"],
  ["spam","Claim your complimentary gift now"],
  ["spam","You have unused reward points click here"],
  ["spam","Huge profits await you"],
  ["ham","Hey are we meeting today"],
  ["ham","Call me when you are free"],
  ["ham","Lets go for lunch tomorrow"],
  ["ham","Did you complete the assignment"],
  ["ham","I will reach by 5 pm"],
  ["ham","Can you send me the notes"],
  ["ham","Lets study together for exams"],
  ["ham","Where are you right now"],
  ["ham","Dont forget the meeting at 3"],
  ["ham","I will call you later"],
  ["ham","Are you coming to the party"],
  ["ham","Please review the document"],
  ["ham","Lets catch up this weekend"],
  ["ham","Meeting has been rescheduled"],
  ["ham","See you in class tomorrow"],
  ["ham","Did you finish the homework"],
  ["ham","What time is the train"],
  ["ham","Can we reschedule for tomorrow"],
  ["ham","I sent you the report"],
  ["ham","Please check your email"],
  ["ham","Can you bring the charger"],
  ["ham","Let me know when you arrive"],
  ["ham","How was your day"],
  ["ham","Are you free on Saturday"],
  ["ham","I will pick you up at noon"],
  ["ham","Thanks for your help yesterday"],
  ["ham","Please confirm attendance"],
  ["ham","Did you get my message"],
  ["ham","See you at the airport"],
  ["ham","Happy birthday have a great day"],
  ["ham","Reminder to submit the form by Friday"],
  ["ham","I need your feedback on this draft"],
  ["ham","Lunch is ready come downstairs"],
  ["ham","Good morning how are you"],
  ["ham","I am running a bit late"],
  ["ham","Can you help me with this problem"],
  ["ham","Should we order pizza or cook dinner"],
  ["ham","Do you want to go for a walk"],
  ["ham","Let me know your availability this week"],
  ["ham","I will be there in ten minutes"],
];

// ─────────────────────────────────────────────────────────────
// preprocess()  –  mirrors SRC/preprocess.py
// Lowercases and strips all non-alpha characters, then splits.
// ─────────────────────────────────────────────────────────────
function preprocess(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z\s]/g, '')  // remove punctuation, digits
    .trim()
    .split(/\s+/)               // tokenise on whitespace
    .filter(Boolean);           // drop empty strings
}

// ─────────────────────────────────────────────────────────────
// trainModel()  –  mirrors NaiveBayes.train_from_list()
// Builds word-frequency dictionaries and stores class counts.
// ─────────────────────────────────────────────────────────────
function trainModel(data) {
  const spamWords = {};   // word → count in spam messages
  const hamWords  = {};   // word → count in ham  messages
  let spamCount   = 0;    // number of spam training messages
  let hamCount    = 0;    // number of ham  training messages
  const vocab     = new Set();  // all unique words seen

  for (const [label, text] of data) {
    const words = preprocess(text);

    if (label === 'spam') {
      spamCount++;
      for (const w of words) {
        spamWords[w] = (spamWords[w] || 0) + 1;
        vocab.add(w);
      }
    } else {
      hamCount++;
      for (const w of words) {
        hamWords[w] = (hamWords[w] || 0) + 1;
        vocab.add(w);
      }
    }
  }

  // cache denominators (Laplace-smoothed total per class)
  const spamTotal = Object.values(spamWords).reduce((a,b)=>a+b,0) + vocab.size;
  const hamTotal  = Object.values(hamWords ).reduce((a,b)=>a+b,0) + vocab.size;

  return { spamWords, hamWords, spamCount, hamCount,
           vocab, spamTotal, hamTotal,
           total: spamCount + hamCount };
}

// ─────────────────────────────────────────────────────────────
// predict()  –  mirrors SRC/predict.py
// Returns { label, logSpam, logHam, wordScores[] }
// ─────────────────────────────────────────────────────────────
function predict(model, text) {
  const words = preprocess(text);
  if (words.length === 0) return { label:'unknown', wordScores:[] };

  // Log-prior probabilities
  let logSpam = Math.log(model.spamCount / model.total);
  let logHam  = Math.log(model.hamCount  / model.total);

  const wordScores = [];

  // Accumulate log-likelihoods (Laplace smoothed)
  for (const w of words) {
    const spamFreq = (model.spamWords[w] || 0) + 1;
    const hamFreq  = (model.hamWords[w]  || 0) + 1;

    const lSpam = Math.log(spamFreq / model.spamTotal);
    const lHam  = Math.log(hamFreq  / model.hamTotal);

    logSpam += lSpam;
    logHam  += lHam;

    // classify each word individually for the word panel display
    wordScores.push({
      word: w,
      spamScore: lSpam,
      hamScore:  lHam,
      bias: lSpam > lHam ? 'spam' : lHam > lSpam ? 'ham' : 'neutral'
    });
  }

  return {
    label:    logSpam > logHam ? 'spam' : 'ham',
    logSpam,
    logHam,
    wordScores
  };
}

// ── Train the in-browser model on page load ───────────────────
const MODEL = trainModel(DATASET);

// ─────────────────────────────────────────────────────────────
// classify()  –  called by the "Classify Message" button
// Reads the textarea, calls predict(), and renders the result.
// ─────────────────────────────────────────────────────────────
function classify() {
  const text = document.getElementById('msgInput').value.trim();
  if (!text) {
    showResult(null, null, []);
    return;
  }

  // Disable button briefly to show processing
  const btn = document.getElementById('classifyBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Classifying…';

  // Small delay for visual feedback
  setTimeout(() => {
    const result = predict(MODEL, text);
    showResult(result.label, result, result.wordScores);
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">🔍</span> Classify Message';
  }, 400);
}

// ─────────────────────────────────────────────────────────────
// showResult()  –  renders the classification result card
// Also populates the word-analysis chip panel.
// ─────────────────────────────────────────────────────────────
function showResult(label, result, wordScores) {
  const area = document.getElementById('resultArea');
  const panel = document.getElementById('wordPanel');
  const wordList = document.getElementById('wordList');

  if (!label) {
    area.className = 'result-area';
    area.innerHTML = `<div class="result-placeholder">
      <span class="result-placeholder-icon">💬</span>
      <span>Enter a message above to classify it</span>
    </div>`;
    panel.classList.add('hidden');
    return;
  }

  const isSpam   = label === 'spam';
  const emoji    = isSpam ? '🚨' : '✅';
  const labelTxt = isSpam ? 'SPAM' : 'HAM (Safe)';
  const msg      = isSpam
    ? 'This message shows strong spam signals.'
    : 'This message looks like a normal conversation.';

  // Update result area
  area.className = `result-area result-${label}`;
  area.innerHTML = `
    <div>
      <div class="result-label result-label-${label}">${emoji} ${labelTxt}</div>
      <div class="result-confidence">${msg}</div>
    </div>
  `;

  // Populate word chips
  wordList.innerHTML = '';
  for (const ws of wordScores) {
    const chip = document.createElement('div');
    chip.className = `word-chip word-chip-${ws.bias}`;
    chip.title = `spam score: ${ws.spamScore.toFixed(3)}  |  ham score: ${ws.hamScore.toFixed(3)}`;
    chip.innerHTML = `${ws.word} <small>${ws.bias === 'spam' ? '🔴' : ws.bias === 'ham' ? '🟢' : '⚪'}</small>`;
    wordList.appendChild(chip);
  }

  panel.classList.remove('hidden');
}

// ─────────────────────────────────────────────────────────────
// setMsg()  –  fill the textarea with a sample message (called
// by the quick-test sample buttons in the section)
// ─────────────────────────────────────────────────────────────
function setMsg(text) {
  document.getElementById('msgInput').value = text;
  classify();
}

// ─────────────────────────────────────────────────────────────
// loadExample()  –  pick a random example from the dataset
// ─────────────────────────────────────────────────────────────
function loadExample() {
  const idx  = Math.floor(Math.random() * DATASET.length);
  const [, txt] = DATASET[idx];
  document.getElementById('msgInput').value = txt;
  classify();
}

// ─────────────────────────────────────────────────────────────
// Enter key in textarea triggers classify
// ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('msgInput')
    .addEventListener('keydown', e => {
      if (e.key === 'Enter' && e.ctrlKey) classify();  // Ctrl+Enter
    });
});

// END OF FILE
// trainModel()  –  builds word-frequency tables from the embedded dataset
// predict()     –  log-probability Naive Bayes (mirrors SRC/predict.py exactly)
// classify()    –  button handler: reads textarea, runs predict, renders result
// showResult()  –  updates the result card and word-analysis panel
// setMsg()      –  fills textarea from quick-test sample buttons
// loadExample() –  picks a random training example for demonstration
