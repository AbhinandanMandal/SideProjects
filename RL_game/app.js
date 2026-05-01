const canvas = document.querySelector("#maze");
const ctx = canvas.getContext("2d");

const ui = {
  episode: document.querySelector("#episode"),
  steps: document.querySelector("#steps"),
  reward: document.querySelector("#reward"),
  epsilon: document.querySelector("#epsilon"),
  train: document.querySelector("#train"),
  watch: document.querySelector("#watch"),
  step: document.querySelector("#step"),
  reset: document.querySelector("#reset"),
  speed: document.querySelector("#speed"),
};

const maze = [
  "S....#....",
  ".##..#.##.",
  "...#...#..",
  "##.#.###.#",
  "...#.....#",
  ".#####.#..",
  ".#.....#.#",
  ".#.###...#",
  "...#..##..",
  "##...#...G",
];

const rows = maze.length;
const cols = maze[0].length;
const start = findTile("S");
const goal = findTile("G");
const actions = [
  { name: "up", dr: -1, dc: 0 },
  { name: "right", dr: 0, dc: 1 },
  { name: "down", dr: 1, dc: 0 },
  { name: "left", dr: 0, dc: -1 },
];

const config = {
  alpha: 0.2,
  gamma: 0.92,
  minEpsilon: 0.04,
  epsilonDecay: 0.992,
  maxSteps: 180,
};

let qTable;
let episode;
let epsilon;
let current;
let steps;
let totalReward;
let trainingTimer = null;
let watchingTimer = null;

function findTile(tile) {
  for (let r = 0; r < rows; r += 1) {
    const c = maze[r].indexOf(tile);
    if (c !== -1) return { r, c };
  }
  throw new Error(`Tile ${tile} not found`);
}

function key(state) {
  return `${state.r},${state.c}`;
}

function createQTable() {
  const table = {};
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      if (maze[r][c] !== "#") table[key({ r, c })] = [0, 0, 0, 0];
    }
  }
  return table;
}

function resetLearning() {
  stopTimers();
  qTable = createQTable();
  episode = 0;
  epsilon = 1;
  resetEpisode();
  render();
}

function resetEpisode() {
  current = { ...start };
  steps = 0;
  totalReward = 0;
  updateStats();
}

function isWall(state) {
  return (
    state.r < 0 ||
    state.c < 0 ||
    state.r >= rows ||
    state.c >= cols ||
    maze[state.r][state.c] === "#"
  );
}

function isGoal(state) {
  return state.r === goal.r && state.c === goal.c;
}

function chooseAction(state, exploring = true) {
  if (exploring && Math.random() < epsilon) {
    return Math.floor(Math.random() * actions.length);
  }

  const values = qTable[key(state)];
  const bestValue = Math.max(...values);
  const bestActions = values
    .map((value, index) => (value === bestValue ? index : -1))
    .filter((index) => index !== -1);

  return bestActions[Math.floor(Math.random() * bestActions.length)];
}

function move(state, actionIndex) {
  const action = actions[actionIndex];
  const next = { r: state.r + action.dr, c: state.c + action.dc };

  if (isWall(next)) {
    return { next: state, reward: -8, done: false };
  }

  if (isGoal(next)) {
    const speedBonus = Math.max(0, config.maxSteps - steps);
    return { next, reward: 100 + speedBonus, done: true };
  }

  return { next, reward: -1, done: false };
}

function learnStep() {
  const actionIndex = chooseAction(current);
  const result = move(current, actionIndex);
  const currentQ = qTable[key(current)][actionIndex];
  const bestFutureQ = Math.max(...qTable[key(result.next)]);
  const target = result.reward + config.gamma * bestFutureQ;

  qTable[key(current)][actionIndex] =
    currentQ + config.alpha * (target - currentQ);

  current = result.next;
  steps += 1;
  totalReward += result.reward;

  if (result.done || steps >= config.maxSteps) {
    episode += 1;
    epsilon = Math.max(config.minEpsilon, epsilon * config.epsilonDecay);
    resetEpisode();
  }

  updateStats();
  render();
}

function trainEpisodes(count) {
  stopTimers();
  let remaining = count;

  trainingTimer = setInterval(() => {
    const loops = Number(ui.speed.value);
    for (let i = 0; i < loops && remaining > 0; i += 1) {
      const before = episode;
      while (episode === before) learnStep();
      remaining -= 1;
    }

    render();
    if (remaining <= 0) stopTimers();
  }, 16);
}

function watchPolicy() {
  stopTimers();
  current = { ...start };
  steps = 0;
  totalReward = 0;

  watchingTimer = setInterval(() => {
    const actionIndex = chooseAction(current, false);
    const result = move(current, actionIndex);
    current = result.next;
    steps += 1;
    totalReward += result.reward;
    updateStats();
    render();

    if (result.done || steps >= config.maxSteps) stopTimers();
  }, 120);
}

function stopTimers() {
  clearInterval(trainingTimer);
  clearInterval(watchingTimer);
  trainingTimer = null;
  watchingTimer = null;
}

function updateStats() {
  ui.episode.textContent = episode;
  ui.steps.textContent = steps;
  ui.reward.textContent = Math.round(totalReward);
  ui.epsilon.textContent = epsilon.toFixed(2);
}

function render() {
  const cell = canvas.width / cols;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const x = c * cell;
      const y = r * cell;
      const tile = maze[r][c];

      ctx.fillStyle = tile === "#" ? "#25313d" : "#fbfaf6";
      ctx.fillRect(x, y, cell, cell);
      ctx.strokeStyle = "#ded8ca";
      ctx.strokeRect(x, y, cell, cell);

      if (tile !== "#") drawBestMove({ r, c }, x, y, cell);
      if (tile === "G") drawGoal(x, y, cell);
      if (tile === "S") drawStart(x, y, cell);
    }
  }

  drawAgent(current, cell);
}

function drawBestMove(state, x, y, cell) {
  const values = qTable[key(state)];
  if (!values || Math.max(...values) <= 0) return;

  const actionIndex = values.indexOf(Math.max(...values));
  const action = actions[actionIndex];
  const centerX = x + cell / 2;
  const centerY = y + cell / 2;

  ctx.save();
  ctx.strokeStyle = "rgba(60, 125, 217, 0.42)";
  ctx.lineWidth = 5;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(centerX, centerY);
  ctx.lineTo(centerX + action.dc * cell * 0.26, centerY + action.dr * cell * 0.26);
  ctx.stroke();
  ctx.restore();
}

function drawGoal(x, y, cell) {
  ctx.fillStyle = "#2f9f73";
  ctx.fillRect(x + cell * 0.22, y + cell * 0.22, cell * 0.56, cell * 0.56);
}

function drawStart(x, y, cell) {
  ctx.strokeStyle = "#3c7dd9";
  ctx.lineWidth = 4;
  ctx.strokeRect(x + cell * 0.22, y + cell * 0.22, cell * 0.56, cell * 0.56);
}

function drawAgent(state, cell) {
  ctx.fillStyle = "#e04f3a";
  ctx.beginPath();
  ctx.arc(
    state.c * cell + cell / 2,
    state.r * cell + cell / 2,
    cell * 0.28,
    0,
    Math.PI * 2,
  );
  ctx.fill();
}

ui.train.addEventListener("click", () => trainEpisodes(500));
ui.watch.addEventListener("click", watchPolicy);
ui.step.addEventListener("click", learnStep);
ui.reset.addEventListener("click", resetLearning);

resetLearning();
