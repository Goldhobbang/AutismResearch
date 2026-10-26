/* 시연용 Mock 데이터 — 실제 측정·AI 추론 없음 */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const app = $('#app');
  const TICK = 700;          // ms per sample
  const N = 36;              // samples shown in chart
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const rnd = a => (Math.random() * 2 - 1) * a;
  const nowStr = () => { const d = new Date(); return `${d.getHours()}:${String(d.getMinutes()).padStart(2, '0')}`; };
  const PROFILES = [
    { id: 'kim-seonrin', name: '김선린', className: '2학년 5반', baseline: { hr: 80, eda: 1.6, risk: 17, mv: .12 }, previous: { hr: 96, eda: 2.8, risk: 52 } },
    { id: 'lee-gyeongsan', name: '이경산', className: '2학년 3반', baseline: { hr: 88, eda: 2.0, risk: 24, mv: .18 }, previous: { hr: 101, eda: 3.1, risk: 47 } },
    { id: 'kang-daegu', name: '강대구', className: '3학년 1반', baseline: { hr: 74, eda: 1.3, risk: 12, mv: .10 }, previous: { hr: 91, eda: 2.4, risk: 39 } }
  ];
  const profileById = new Map(PROFILES.map(profile => [profile.id, profile]));
  const studentName = () => profileById.get(S.profileId).name;
  const studentClass = () => profileById.get(S.profileId).className;

  /* ---------- Scenarios: target values as function of elapsed seconds ---------- */
  const ramp = (t, a, b, d) => a + (b - a) * Math.min(1, Math.max(0, t / d));
  const SCN = {
    calm: {
      dur: 30, state: 'safe', ctx: '교실 · 수업 중',
      at: (t, baseline) => ({ ...baseline }),
      noise: { hr: 2.6, eda: .14, risk: 3, mv: .04 },
      msg: '평소 범위 안에서 안정적으로 유지되고 있어요.',
      insight: '최근 30초간 심박수와 피부전도가 개인 기준선 범위 안에 있습니다.'
    },
    stress: {
      dur: 40, ctx: '교실 · 소음 증가',
      at: (t, b) => ({ hr: ramp(t, b.hr, b.hr + 33, 18), eda: ramp(t, b.eda, b.eda + 2.6, 18), risk: ramp(t, b.risk, 90, 18), mv: ramp(t, b.mv, b.mv + .1, 18) }),
      noise: { hr: 2.8, eda: .18, risk: 2.2, mv: .04 },
      alertAt: 17
    },
    exercise: {
      dur: 40, ctx: '운동장 · 체육 활동',
      at: (t, b) => ({ hr: ramp(t, b.hr, b.hr + 48, 14), eda: ramp(t, b.eda, b.eda + .8, 14), risk: ramp(t, b.risk, b.risk + 10, 14), mv: ramp(t, b.mv, .9, 10) }),
      noise: { hr: 4, eda: .12, risk: 3, mv: .08 },
      alertAt: 12
    }
  };

  function seed(profile) {
    return [{
      id: 0, level: 'warn', title: '주의 상태 감지', time: '어제 14:08', ack: true,
      body: `${profile.name} 학생의 점심시간 이후 심박수가 일시적으로 올랐다가 기준선으로 돌아왔습니다.`,
      ...profile.previous, ackTime: '어제 14:12'
    }];
  }
  function createProfileState(profile) {
    return {
      profileId: profile.id, baseline: { ...profile.baseline },
      scn: 'calm', t: 0,
      cur: { ...profile.baseline },
      hrH: Array.from({ length: N }, () => profile.baseline.hr + rnd(2.5)),
      edaH: Array.from({ length: N }, () => profile.baseline.eda + rnd(.12)),
      alerts: seed(profile), filter: 'all', fired: false, uid: 1, openId: null
    };
  }
  const profileStates = new Map(PROFILES.map(profile => [profile.id, createProfileState(profile)]));
  let S = profileStates.get(PROFILES[0].id);

  /* ---------- Level logic ---------- */
  function level() {
    if (S.scn === 'exercise' && S.cur.mv > .45) return 'exercise';
    const r = S.cur.risk;
    return r >= 70 ? 'risk' : r >= 40 ? 'warn' : 'safe';
  }
  const TXT = {
    safe: ['안정', '평소 범위 안에서 안정적으로 유지되고 있어요.'],
    warn: ['주의', '심박수와 피부전도가 평소보다 오르고 있어요. 상태를 지켜보는 중입니다.'],
    risk: ['위험', '스트레스 관련 신호가 평소보다 높게 나타났어요. 지금 확인해 주세요.'],
    exercise: ['운동 중', '움직임이 함께 늘어 운동에 의한 변화로 분석했어요.']
  };

  /* ---------- Tick ---------- */
  function sample() {
    const sc = SCN[S.scn], tg = sc.at(S.t, S.baseline), n = sc.noise, c = S.cur;
    for (const k of ['hr', 'eda', 'risk', 'mv']) {
      c[k] += (tg[k] - c[k]) * .3 + rnd(n[k]) * .6;
    }
    c.hr = clamp(c.hr, 60, 150); c.eda = clamp(c.eda, .6, 6); c.risk = clamp(c.risk, 3, 99); c.mv = clamp(c.mv, 0, 1);
    S.hrH.push(c.hr); S.edaH.push(c.eda);
    if (S.hrH.length > N) { S.hrH.shift(); S.edaH.shift(); }
    S.t += TICK / 1000;

    if (!S.fired && sc.alertAt && S.t >= sc.alertAt) {
      S.fired = true;
      if (S.scn === 'stress') raise('risk');
      if (S.scn === 'exercise') raise('info');
    }
    render();
  }

  /* ---------- Rendering ---------- */
  function render() {
    const c = S.cur, lv = level();
    app.dataset.state = lv;
    $('#studentName').textContent = studentName();
    $('#studentClass').textContent = studentClass();
    const [label, msg] = TXT[lv];
    $('#stateText').textContent = label;
    $('#heroMsg').textContent = msg;
    $('#ctx').textContent = SCN[S.scn].ctx;
    $('#riskNum').textContent = Math.round(c.risk);
    $('#ringFg').style.strokeDashoffset = 100 - c.risk;
    $('#hrNum').textContent = Math.round(c.hr);
    $('#edaNum').textContent = c.eda.toFixed(1);
    $('#hrSub').textContent = c.hr < 90 ? '평소 범위' : c.hr < 105 ? '조금 높음' : '높음';
    $('#edaSub').textContent = c.eda < 2.1 ? '낮음' : c.eda < 3 ? '보통' : '높음';
    $('#mvText').textContent = c.mv < .3 ? '낮음' : c.mv < .6 ? '보통' : '활발함';
    $$('#mvBars i').forEach((b, i) => b.classList.toggle('on', c.mv * 5 > i + .3));

    let ins;
    if (lv === 'exercise') ins = '심박수는 높지만 움직임이 함께 증가했습니다. 운동에 의한 변화로 판단되어 위험 알림을 보내지 않았습니다.';
    else if (lv === 'risk') ins = '움직임 변화 없이 심박수와 피부전도가 함께 상승했습니다. 스트레스 관련 신호일 가능성이 높아 알림을 보냈습니다.';
    else if (lv === 'warn') ins = '움직임은 적은데 심박수와 피부전도가 오르고 있습니다. 추이를 계속 분석하는 중입니다.';
    else ins = '최근 30초간 심박수와 피부전도가 개인 기준선 범위 안에 있습니다.';
    $('#insight').textContent = ins;

    drawChart();
    const sc = SCN[S.scn];
    $('#runT').textContent = `${Math.floor(S.t / 60)}:${String(Math.floor(S.t % 60)).padStart(2, '0')}`;
    $('#progBar').style.width = Math.min(100, S.t / sc.dur * 100) + '%';
  }

  function path(vals, lo, hi) {
    const W = 340, H = 150, pad = 14, step = W / (N - 1), off = N - vals.length;
    const pts = vals.map((v, i) => [(i + off) * step, H - pad - (clamp(v, lo, hi) - lo) / (hi - lo) * (H - pad * 2)]);
    let d = `M${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`;
    for (let i = 1; i < pts.length; i++) {
      const [x0, y0] = pts[i - 1], [x1, y1] = pts[i], mx = (x0 + x1) / 2;
      d += `C${mx.toFixed(1)} ${y0.toFixed(1)} ${mx.toFixed(1)} ${y1.toFixed(1)} ${x1.toFixed(1)} ${y1.toFixed(1)}`;
    }
    return { d, pts };
  }
  function drawChart() {
    const h = path(S.hrH, 60, 150), e = path(S.edaH, 0.5, 6);
    $('#lineHr').setAttribute('d', h.d);
    $('#lineEda').setAttribute('d', e.d);
    const last = h.pts[h.pts.length - 1], first = h.pts[0];
    $('#areaHr').setAttribute('d', `${h.d}L${last[0]} 150L${first[0]} 150Z`);
    const dot = $('#dotHr');
    dot.style.left = (last[0] / 340 * 100).toFixed(2) + '%';
    dot.style.top = (last[1] / 150 * 100).toFixed(2) + '%';
  }

  /* ---------- Alerts ---------- */
  const COPY = {
    risk: { title: '위험 징후 감지', body: name => `${name} 학생의 스트레스 관련 생체신호가 평소보다 높게 나타났습니다.`, body2: name => `${name} 학생의 상태를 확인하고, 필요하면 주변 자극을 줄이거나 휴식을 지원해 주세요.`, lbl: '위험' },
    warn: { title: '주의 상태 감지', body: name => `${name} 학생의 심박수와 피부전도가 평소보다 올라가고 있습니다.`, body2: name => `${name} 학생의 상태를 한 번 살펴봐 주세요.`, lbl: '주의' },
    info: { title: '운동으로 인한 변화', body: '심박수가 올랐지만 움직임이 함께 증가해 운동에 의한 변화로 분석했습니다.', body2: '위험 알림은 보내지 않았고, 이 기록은 참고용입니다.', lbl: '참고' }
  };
  function raise(lv) {
    const c = S.cur, cp = COPY[lv], name = studentName();
    const a = {
      id: S.uid++, level: lv, title: cp.title, time: '방금', clock: nowStr(), ack: false,
      body: typeof cp.body === 'function' ? cp.body(name) : cp.body,
      body2: typeof cp.body2 === 'function' ? cp.body2(name) : cp.body2,
      hr: Math.round(c.hr), eda: +c.eda.toFixed(1), risk: Math.round(c.risk)
    };
    a.time = a.clock;
    S.alerts.unshift(a);
    renderAlerts();
    if (lv === 'risk') showBanner(a.id);
  }
  const ICON = { risk: '!', warn: '!', info: 'i' };
  const LBL = { risk: '위험', warn: '주의', info: '참고' };

  function renderAlerts() {
    const open = S.alerts.filter(a => !a.ack && a.level !== 'info').length;
    $('#openCount').textContent = open;
    const b = $('#tabBadge'); b.textContent = open; b.classList.toggle('show', open > 0);
    const list = S.alerts.filter(a => S.filter === 'all' || (!a.ack && a.level !== 'info'));
    $('#alertList').innerHTML = list.length ? list.map(a => `
      <button class="acard ${a.level} ${a.ack ? 'ack' : ''}" data-id="${a.id}">
        <div class="row"><span class="tag ${a.ack ? 'done' : a.level}">${a.ack ? '확인 완료' : LBL[a.level]}</span><span class="time">${a.time}</span></div>
        <h4>${a.title}</h4><p>${a.body}</p>
        <div class="foot">${a.ack ? '✓ ' + (a.ackTime || '확인함') + ' 확인' : a.level === 'info' ? '알림 억제됨 · 자세히 보기' : '확인이 필요해요 ›'}</div>
      </button>`).join('') : '<p class="empty">확인할 알림이 없어요.</p>';

    const rec = S.alerts.slice(0, 2);
    $('#recentList').innerHTML = rec.map(a => `
      <button class="al ${a.level}" data-id="${a.id}">
        <span class="al-ico">${ICON[a.level]}</span>
        <span class="al-t"><b>${a.title}</b><small>${a.body}</small></span>
        <span class="al-r"><span>${a.time}</span><span class="tag ${a.ack ? 'done' : a.level}">${a.ack ? '확인함' : LBL[a.level]}</span></span>
      </button>`).join('');
  }

  /* ---------- Banner & Sheet ---------- */
  let bannerT, bannerId = null;
  function showBanner(id) {
    bannerId = id; $('#banner').classList.add('show');
    clearTimeout(bannerT); bannerT = setTimeout(hideBanner, 6000);
  }
  function hideBanner() { $('#banner').classList.remove('show'); }
  $('#banner').onclick = () => { hideBanner(); openSheet(bannerId); };

  function openSheet(id) {
    const a = S.alerts.find(x => x.id === id); if (!a) return;
    S.openId = id;
    const isInfo = a.level === 'info';
    $('#sheetBody').innerHTML = `
      <div class="sh-ico ${a.level}">${ICON[a.level]}</div>
      <h3>${a.title}</h3>
      <p class="body">${a.body}</p>
      ${a.body2 ? `<p class="body">${a.body2}</p>` : ''}
      <div class="stats">
        <div><small>심박수</small><b>${a.hr}</b><em>bpm</em></div>
        <div><small>피부전도</small><b>${a.eda}</b><em>μS</em></div>
        <div><small>위험 점수</small><b>${a.risk}</b><em>/100</em></div>
      </div>
      <p class="meta"><b>${studentName()}</b> · ${studentClass()} · ${a.time}${a.ack ? ' · ' + (a.ackTime || '') + ' 확인' : ''}<br>시연용 Mock 데이터입니다.</p>
      <button class="ackbtn ${a.ack ? 'done' : isInfo ? 'blue' : ''}" id="ackBtn">${a.ack ? '✓ 확인 완료' : isInfo ? '기록 확인' : '확인 완료'}</button>
      <button class="close" id="closeSheet">닫기</button>`;
    $('#ackBtn').onclick = () => ack(id);
    $('#closeSheet').onclick = closeSheet;
    $('#sheet').classList.add('show'); $('#scrim').classList.add('show');
    $('#sheet').setAttribute('aria-hidden', 'false');
  }
  function closeSheet() {
    $('#sheet').classList.remove('show'); $('#scrim').classList.remove('show');
    $('#sheet').setAttribute('aria-hidden', 'true'); S.openId = null;
  }
  $('#scrim').onclick = closeSheet;

  const savePngButton = $('#savePng');
  const savePngStatus = $('#savePngStatus');
  let saveStatusTimer;

  function showSaveStatus(message, isError = false) {
    savePngStatus.textContent = message;
    savePngStatus.classList.toggle('error', isError);
    savePngStatus.classList.add('show');
    clearTimeout(saveStatusTimer);
    saveStatusTimer = setTimeout(() => savePngStatus.classList.remove('show'), 3500);
  }
  savePngButton.onclick = async () => {
    if (typeof window.html2canvas !== 'function') {
      showSaveStatus('PNG 저장 기능을 불러오지 못했어요. 인터넷 연결을 확인해 주세요.', true);
      return;
    }

    savePngButton.disabled = true;
    savePngButton.querySelector('span').textContent = '저장 중';
    try {
      await document.fonts.ready;
      const canvas = await window.html2canvas(app, {
        backgroundColor: null,
        scale: 2,
        useCORS: true,
        logging: false
      });
      const blob = await new Promise((resolve, reject) => {
        canvas.toBlob(result => result ? resolve(result) : reject(new Error('PNG image could not be created')), 'image/png');
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${studentName()}_${studentClass().replaceAll(' ', '')}_마음결.png`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      showSaveStatus(`${studentName()} 화면을 PNG로 저장했어요.`);
    } catch (error) {
      console.error('PNG 저장에 실패했습니다.', error);
      showSaveStatus('PNG 저장에 실패했어요. 다시 시도해 주세요.', true);
    } finally {
      savePngButton.disabled = false;
      savePngButton.querySelector('span').textContent = 'PNG 저장';
    }
  };

  function ack(id) {
    const a = S.alerts.find(x => x.id === id); if (!a || a.ack) return;
    a.ack = true; a.ackTime = nowStr();
    const btn = $('#ackBtn'); btn.textContent = '✓ 확인 완료'; btn.className = 'ackbtn done';
    hideBanner(); renderAlerts();
    setTimeout(() => { if (S.openId === id) closeSheet(); }, 900);
  }
  $('#markAll').onclick = () => { S.alerts.forEach(a => { if (!a.ack) { a.ack = true; a.ackTime = nowStr(); } }); hideBanner(); renderAlerts(); };

  document.addEventListener('click', e => {
    const card = e.target.closest('[data-id]');
    if (card) openSheet(+card.dataset.id);
  });

  /* ---------- Navigation ---------- */
  function go(name) {
    $$('.screen').forEach(s => s.classList.toggle('active', s.id === 's-' + name));
    $$('.tab').forEach(t => t.classList.toggle('on', t.dataset.go === name));
    $$('.scroll').forEach(s => s.scrollTop = 0);
    closeSheet();
  }
  $$('[data-go]').forEach(b => b.addEventListener('click', () => go(b.dataset.go)));
  const profileTrigger = $('#profileTrigger');
  const profileMenu = $('#profileMenu');
  function syncScenarioControls() {
    app.dataset.scn = S.scn;
    $$('.scn').forEach(b => b.classList.toggle('on', b.dataset.scn === S.scn));
  }
  function renderProfileMenu() {
    profileMenu.innerHTML = PROFILES.map(profile => `
      <button class="profile-option" type="button" role="menuitem" data-profile-id="${profile.id}" aria-current="${profile.id === S.profileId}">
        <span class="profile-option-name">${profile.name}</span>
        <span class="profile-option-class">${profile.className}</span>
      </button>`).join('');
  }
  function closeProfileMenu(returnFocus = false) {
    profileMenu.hidden = true;
    profileTrigger.setAttribute('aria-expanded', 'false');
    if (returnFocus) profileTrigger.focus();
  }
  function selectProfile(profileId) {
    if (!profileStates.has(profileId) || profileId === S.profileId) {
      closeProfileMenu(true);
      return;
    }
    clearTimeout(bannerT);
    hideBanner();
    closeSheet();
    closeProfileMenu();
    S = profileStates.get(profileId);
    renderProfileMenu();
    render();
    renderAlerts();
    syncScenarioControls();
    go('home');
    profileTrigger.focus();
  }
  profileTrigger.addEventListener('click', () => {
    const isOpen = profileTrigger.getAttribute('aria-expanded') === 'true';
    profileMenu.hidden = isOpen;
    profileTrigger.setAttribute('aria-expanded', String(!isOpen));
    if (!isOpen) profileMenu.querySelector('[aria-current="true"]')?.focus();
  });
  profileMenu.addEventListener('click', event => {
    const option = event.target.closest('[data-profile-id]');
    if (option) selectProfile(option.dataset.profileId);
  });
  document.addEventListener('click', event => {
    if (!profileMenu.hidden && !event.target.closest('.profile-control')) closeProfileMenu();
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !profileMenu.hidden) closeProfileMenu(true);
  });
  renderProfileMenu();
  $$('#seg button').forEach(b => b.onclick = () => {
    S.filter = b.dataset.f; $$('#seg button').forEach(x => x.classList.toggle('on', x === b)); renderAlerts();
  });

  /* ---------- Scenario control ---------- */
  function setScenario(key) {
    S.scn = key; S.t = 0; S.fired = false;
    syncScenarioControls();
    render();
  }
  $$('.scn').forEach(b => b.onclick = () => { setScenario(b.dataset.scn); });
  $('#replay').onclick = () => {
    // 현재 시나리오를 안정 상태에서 다시 시작
    S.cur = { ...S.baseline }; S.hrH = []; S.edaH = [];
    prefill(); setScenario(S.scn);
  };
  $('#resetAll').onclick = () => { S.alerts = seed(profileById.get(S.profileId)); S.uid = 1; hideBanner(); renderAlerts(); };

  function prefill() {
    for (let i = 0; i < N; i++) { S.hrH.push(S.baseline.hr + rnd(2.5)); S.edaH.push(S.baseline.eda + rnd(.12)); }
  }

  /* ---------- Boot ---------- */
  render(); renderAlerts(); setScenario('calm');
  let simulationTimer = setInterval(sample, TICK);
  document.addEventListener('visibilitychange', () => {
    clearInterval(simulationTimer);
    if (!document.hidden) simulationTimer = setInterval(sample, TICK);
  });
})();
