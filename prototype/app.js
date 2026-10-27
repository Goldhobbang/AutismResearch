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
    { id: 'kim-seonrin', name: '김선린', info: '2학년 5반', baseline: { hr: 80, eda: 1.6, risk: 17, mv: .12 }, previous: { hr: 96, eda: 2.8, risk: 52 } },
    { id: 'lee-gyeongsan', name: '이경산', info: '2학년 3반', baseline: { hr: 88, eda: 2.0, risk: 24, mv: .18 }, previous: { hr: 101, eda: 3.1, risk: 47 } },
    { id: 'kang-daegu', name: '강대구', info: '3학년 1반', baseline: { hr: 74, eda: 1.3, risk: 12, mv: .10 }, previous: { hr: 91, eda: 2.4, risk: 39 } }
  ];
  const profileById = new Map(PROFILES.map(profile => [profile.id, profile]));
  const studentName = () => profileById.get(S.profileId).name;
  const studentInfo = () => profileById.get(S.profileId).info;

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
      id: 0, level: 'warn', title: '주의 상태 감지', time: '어제 14:08', ack: true, pid: profile.id,
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
      alerts: profile.id === PROFILES[0].id ? seed(profile) : [], filter: 'all', fired: false, uid: 1, openId: null
    };
  }
  function addProfile(name, info, baseline) {
    const base = baseline || { hr: 76 + Math.random() * 12, eda: 1.4 + Math.random() * 0.6, risk: 14 + Math.random() * 10, mv: .1 + Math.random() * .08 };
    const id = 'p' + Date.now().toString(36) + Math.floor(Math.random() * 999);
    const profile = { id, name, info: info || '정보 없음', baseline: { ...base }, previous: { hr: Math.round(base.hr) + 16, eda: +(base.eda + 1.2).toFixed(1), risk: Math.round(base.risk) + 30 } };
    PROFILES.push(profile);
    profileById.set(id, profile);
    profileStates.set(id, createProfileState(profile));
    renderProfileMenu();
    renderOverview();
    return id;
  }
  const profileStates = new Map(PROFILES.map(profile => [profile.id, createProfileState(profile)]));
  let S = profileStates.get(PROFILES[0].id);

  /* ---------- Level logic ---------- */
  function level() {
    return levelOf(S);
  }
  const TXT = {
    safe: ['안정', '평소 범위 안에서 안정적으로 유지되고 있어요.'],
    warn: ['주의', '심박수와 피부전도가 평소보다 오르고 있어요. 상태를 지켜보는 중입니다.'],
    risk: ['위험', '스트레스 관련 신호가 평소보다 높게 나타났어요. 지금 확인해 주세요.'],
    exercise: ['운동 중', '움직임이 함께 늘어 운동에 의한 변화로 분석했어요.']
  };

  // 모든 대상자의 시뮬레이션을 항상 함께 전진시킨다 (화면과 무관하게 진행)
  function tickAll() {
    for (const st of profileStates.values()) tickOne(st);
    if ($('#s-all').classList.contains('active')) renderOverview();
    else if ($('#s-demo').classList.contains('active')) renderPeopleChips();
    else renderSharedAlertsWatch();
  }
  function tickOne(st) {
    const sc = SCN[st.scn], tg = sc.at(st.t, st.baseline), n = sc.noise, c = st.cur;
    for (const k of ['hr', 'eda', 'risk', 'mv']) {
      c[k] += (tg[k] - c[k]) * .3 + rnd(n[k]) * .6;
    }
    c.hr = clamp(c.hr, 60, 150); c.eda = clamp(c.eda, .6, 6); c.risk = clamp(c.risk, 3, 99); c.mv = clamp(c.mv, 0, 1);
    st.hrH.push(c.hr); st.edaH.push(c.eda);
    if (st.hrH.length > N) { st.hrH.shift(); st.edaH.shift(); }
    st.t += TICK / 1000;

    if (!st.fired && sc.alertAt && st.t >= sc.alertAt) {
      st.fired = true;
      if (st.scn === 'stress') raiseFor(st, 'risk');
      if (st.scn === 'exercise') raiseFor(st, 'info');
    }
    if (st === S) render();
  }
  function levelOf(st) {
    if (st.scn === 'exercise' && st.cur.mv > .45) return 'exercise';
    const r = st.cur.risk;
    return r >= 70 ? 'risk' : r >= 40 ? 'warn' : 'safe';
  }

  /* ---------- Rendering ---------- */
  const OVER_N = 48; // 전체 화면용 미니 그래프 샘플 수
  function renderOverview() {
    const el = $('#allList');
    const lvs = PROFILES.map(p => levelOf(profileStates.get(p.id)));
    $('#allCount').textContent = `${PROFILES.length}명 연결됨`;
    $('#sumSafe').textContent = lvs.filter(v => v === 'safe').length;
    $('#sumWarn').textContent = lvs.filter(v => v === 'warn').length;
    $('#sumRisk').textContent = lvs.filter(v => v === 'risk').length;
    $('#sumExercise').textContent = lvs.filter(v => v === 'exercise').length;

    if (el.childElementCount !== PROFILES.length) {
      el.innerHTML = PROFILES.map(p => `
        <button class="all-row" data-profile="${p.id}" type="button">
          <span class="all-dot"></span>
          <span class="all-t"><b>${p.name}</b><small>${p.info}</small></span>
          <span class="all-graph"><svg viewBox="0 0 100 30" preserveAspectRatio="none" aria-hidden="true"><polyline/></svg></span>
          <span class="all-num"><b>0</b><small>위험 점수</small></span>
        </button>`).join('');
    }
    PROFILES.forEach((p, i) => {
      const st = profileStates.get(p.id), lv = lvs[i];
      const row = el.children[i];
      row.className = `all-row lv-${lv}`;
      row.dataset.profile = p.id;
      row.querySelector('.all-t b').textContent = p.name;
      row.querySelector('.all-t small').textContent = p.info;
      row.querySelector('.all-num b').textContent = Math.round(st.cur.risk);
      row.querySelector('.all-num small').textContent = LVL_TXT[lv];
      row.querySelector('polyline').setAttribute('points', overviewPoints(st));
    });
    renderPeopleChips();
  }
  function overviewPoints(st) {
    const vals = st.hrH.slice(-OVER_N);
    const lo = 60, hi = 150;
    return vals.map((v, i) => `${(i / (OVER_N - 1) * 100).toFixed(1)},${(28 - clamp((v - lo) / (hi - lo), 0, 1) * 24).toFixed(1)}`).join(' ');
  }
  function renderPeopleChips() {
    const box = $('#peopleChips');
    if (!box) return;
    $('#regCount').textContent = `${PROFILES.length}명`;
    box.innerHTML = PROFILES.map(p => {
      const lv = levelOf(profileStates.get(p.id));
      return `<span class="people-chip lv-${lv}"><i></i>${p.name}</span>`;
    }).join('');
  }
  document.addEventListener('click', e => {
    const row = e.target.closest('[data-profile]');
    if (row) selectProfile(row.dataset.profile);
  });
  function render() {
    const c = S.cur, lv = level();
    app.dataset.state = lv;
    $('#studentName').textContent = studentName();
    $('#studentClass').textContent = studentInfo();
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
  function raise(lv) { raiseFor(S, lv); }
  function raiseFor(st, lv) {
    const profile = profileById.get(st.profileId);
    const name = profile.name;
    const c = st.cur, cp = COPY[lv];
    const a = {
      id: st.uid++, level: lv, title: cp.title, time: '방금', clock: nowStr(), ack: false, pid: st.profileId,
      body: typeof cp.body === 'function' ? cp.body(name) : cp.body,
      body2: typeof cp.body2 === 'function' ? cp.body2(name) : cp.body2,
      hr: Math.round(c.hr), eda: +c.eda.toFixed(1), risk: Math.round(c.risk)
    };
    a.time = a.clock;
    _push(st, a);
    refreshAfterRaise(a, lv);
  }
  const ICON = { risk: '!', warn: '!', info: 'i' };
  const LBL = { risk: '위험', warn: '주의', info: '참고' };
  const LVL_TXT = { safe: '안정', warn: '주의', risk: '위험', exercise: '운동 중' };
  let alertFilter = 'all';
  let openSeq = null;
  function ownerName(a) { const p = profileById.get(a.pid); return p ? p.name : ''; }
  // 모든 대상자의 알림을 시간순으로 합친다
  function allAlerts() {
    const all = [];
    for (const st of profileStates.values()) for (const a of st.alerts) all.push(a);
    return all.sort((x, y) => y.seq - x.seq);
  }
  let alertSeq = 0;
  const _push = (st, a) => { a.seq = ++alertSeq; st.alerts.unshift(a); };
  // 부트 시 시드 알림에도 순서 번호 부여
  for (const st of profileStates.values()) for (const a of st.alerts) a.seq = ++alertSeq;

  function renderAlerts() {
    const all = allAlerts();
    const open = all.filter(a => !a.ack && a.level !== 'info').length;
    $('#openCount').textContent = open;
    $('#openCount').classList.toggle('zero', open === 0);
    const b = $('#tabBadge'); b.textContent = open; b.classList.toggle('show', open > 0);
    const list = all.filter(a => alertFilter === 'all' || (!a.ack && a.level !== 'info'));
    $('#alertList').innerHTML = list.length ? list.map(a => `
      <button class="acard ${a.level} ${a.ack ? 'ack' : ''}" data-id="${a.seq}">
        <div class="row"><span class="tag ${a.ack ? 'done' : a.level}">${a.ack ? '확인 완료' : LBL[a.level]}</span><span class="time">${ownerName(a)} · ${a.time}</span></div>
        <h4>${a.title}</h4><p>${a.body}</p>
        <div class="foot">${a.ack ? '✓ ' + (a.ackTime || '확인함') + ' 확인' : a.level === 'info' ? '알림 억제됨 · 자세히 보기' : '확인이 필요해요 ›'}</div>
      </button>`).join('') : '<p class="empty">확인할 알림이 없어요.</p>';

    const rec = all.slice(0, 2);
    $('#recentList').innerHTML = rec.map(a => `
      <button class="al ${a.level}" data-id="${a.seq}">
        <span class="al-ico">${ICON[a.level]}</span>
        <span class="al-t"><b>${a.title}</b><small>${ownerName(a)} · ${a.body}</small></span>
        <span class="al-r"><span>${a.time}</span><span class="tag ${a.ack ? 'done' : a.level}">${a.ack ? '확인함' : LBL[a.level]}</span></span>
      </button>`).join('');
  }
  // 알림 화면이 열려 있을 때만 갱신 (전체 화면 틱에서 호출)
  function renderSharedAlertsWatch() {
    if ($('#s-alerts').classList.contains('active')) renderAlerts();
  }

  function findAlert(seq) {
    return allAlerts().find(x => x.seq === seq);
  }
  function refreshAfterRaise(a, lv) {
    renderAlerts();
    if ($('#s-all').classList.contains('active')) renderOverview();
    if (lv === 'risk') {
      const b = $('#banner b'); if (b) b.textContent = `${ownerName(a)} · 위험 징후 감지`;
      showBanner(a.seq);
    }
  }

  /* ---------- Banner & Sheet ---------- */
  let bannerT, bannerId = null;
  function showBanner(id) {
    bannerId = id; $('#banner').classList.add('show');
    clearTimeout(bannerT); bannerT = setTimeout(hideBanner, 6000);
  }
  function hideBanner() { $('#banner').classList.remove('show'); }
  $('#banner').onclick = () => { hideBanner(); openSheet(bannerId); };

  function openSheet(seq) {
    const a = findAlert(seq); if (!a) return;
    openSeq = seq;
    hideToast();
    hideBanner();
    const person = profileById.get(a.pid);
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
      <p class="meta"><b>${person ? person.name : ''}</b> · ${person ? person.info : ''} · ${a.time}${a.ack ? ' · ' + (a.ackTime || '') + ' 확인' : ''}<br>시연용 Mock 데이터입니다.</p>
      <button class="ackbtn ${a.ack ? 'done' : isInfo ? 'blue' : ''}" id="ackBtn">${a.ack ? '✓ 확인 완료' : isInfo ? '기록 확인' : '확인 완료'}</button>
      <button class="close" id="closeSheet">닫기</button>`;
    $('#ackBtn').onclick = () => ack(seq);
    $('#closeSheet').onclick = closeSheet;
    $('#sheet').classList.add('show'); $('#scrim').classList.add('show');
    $('#sheet').setAttribute('aria-hidden', 'false');
  }
  function closeSheet() {
    $('#sheet').classList.remove('show'); $('#scrim').classList.remove('show');
    $('#sheet').setAttribute('aria-hidden', 'true'); openSeq = null;
  }
  $('#scrim').onclick = () => {
    if (regSheet.classList.contains('show')) closeRegSheet();
    else closeSheet();
  };

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
      link.download = `${studentName()}_${studentInfo().replaceAll(' ', '')}_마음결.png`;
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

  function ack(seq) {
    const a = findAlert(seq); if (!a || a.ack) return;
    a.ack = true; a.ackTime = nowStr();
    const btn = $('#ackBtn'); btn.textContent = '✓ 확인 완료'; btn.className = 'ackbtn done';
    hideBanner(); renderAlerts();
    setTimeout(() => { if (openSeq === seq) closeSheet(); }, 900);
  }
  $('#markAll').onclick = () => { for (const a of allAlerts()) { if (!a.ack) { a.ack = true; a.ackTime = nowStr(); } } hideBanner(); renderAlerts(); };

  document.addEventListener('click', e => {
    const card = e.target.closest('[data-id]');
    if (card && !e.target.closest('[data-profile]')) openSheet(+card.dataset.id);
  });

  /* ---------- Navigation ---------- */
  function go(name) {
    $$('.screen').forEach(s => s.classList.toggle('active', s.id === 's-' + name));
    $$('.tab').forEach(t => t.classList.toggle('on', t.dataset.go === name));
    $$('.scroll').forEach(s => s.scrollTop = 0);
    closeSheet();
    hideToast();
  }
  $$('[data-go]').forEach(b => b.addEventListener('click', () => go(b.dataset.go)));
  const profileTrigger = $('#profileTrigger');
  const profileMenu = $('#profileMenu');
  function syncScenarioControls() {
    app.dataset.scn = S.scn;
    $$('.scn').forEach(b => {
      const on = randomMode ? b.dataset.scn === 'random' : b.dataset.scn === S.scn;
      b.classList.toggle('on', on);
    });
  }
  function renderProfileMenu() {
    profileMenu.innerHTML = PROFILES.map(profile => `
      <button class="profile-option" type="button" role="menuitem" data-profile-id="${profile.id}" aria-current="${profile.id === S.profileId}">
        <span class="profile-option-name">${profile.name}</span>
        <span class="profile-option-class">${profile.info}</span>
      </button>`).join('') + `
      <button class="profile-option add" type="button" role="menuitem" data-add-person>
        <span class="profile-option-name">+ 새로운 사람 등록</span>
      </button>`;
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
    renderOverview();
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
    if (event.target.closest('[data-add-person]')) { openRegSheet(); return; }
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
    alertFilter = b.dataset.f; $$('#seg button').forEach(x => x.classList.toggle('on', x === b)); renderAlerts();
  });

  /* ---------- Scenario control ---------- */
  function setScenario(key) {
    S.scn = key; S.t = 0; S.fired = false; randomMode = false;
    syncScenarioControls();
    render();
  }
  $$('.scn').forEach(b => b.onclick = () => { if (b.dataset.scn === 'random') randomizeAll(); else setScenario(b.dataset.scn); });
  $('#replay').onclick = () => {
    // 현재 시나리오를 안정 상태에서 다시 시작
    S.cur = { ...S.baseline }; S.hrH = []; S.edaH = [];
    prefill(); setScenario(S.scn);
  };
  $('#resetAll').onclick = () => {
    for (const [pid, st] of profileStates) {
      st.alerts = seed(profileById.get(pid));
      st.uid = 1;
      for (const a of st.alerts) a.seq = ++alertSeq;
    }
    hideBanner(); renderAlerts();
    if ($('#s-all').classList.contains('active')) renderOverview();
  };
  // 무작위 상태: 등록된 모든 사람에게 안정/스트레스/운동 중 랜덤 배정
  let randomMode = false;
  function randomizeAll() {
    const keys = ['calm', 'stress', 'exercise'];
    let i = 0;
    for (const st of profileStates.values()) {
      const key = keys[Math.floor(Math.random() * 3)];
      st.scn = key;
      st.fired = true;                 // 무작위 배치 중 알림이 한꺼번에 쏟아지지 않게
      // 결과가 바로 보이도록 각 상태의 정상 구간으로 이동
      st.t = key === 'calm' ? 12 : 22;
      const tg = SCN[key].at(st.t, st.baseline);
      st.cur = { ...tg };
      st.hrH = Array.from({ length: N }, (_, k) => {
        const p = k / (N - 1);
        if (key === 'stress') return st.baseline.hr + (tg.hr - st.baseline.hr) * Math.min(1, p * 1.6) + rnd(1.8);
        if (key === 'exercise') return st.baseline.hr + (tg.hr - st.baseline.hr) * Math.min(1, p * 2.2) + rnd(4.5);
        return tg.hr + rnd(2.4);
      });
      st.edaH = Array.from({ length: N }, (_, k) => {
        const p = k / (N - 1);
        if (key === 'stress') return st.baseline.eda + (tg.eda - st.baseline.eda) * Math.min(1, p * 1.6) + rnd(.1);
        if (key === 'exercise') return st.baseline.eda + (tg.eda - st.baseline.eda) * Math.min(1, p * 2.2) + rnd(.1);
        return tg.eda + rnd(.1);
      });
      i++;
    }
    randomMode = true;
    syncScenarioControls();
    render(); renderAlerts(); renderOverview();
    toast(`등록된 ${PROFILES.length}명의 상태를 무작위로 배정했어요`);
  }

  /* ---------- Register sheet ---------- */
  const regSheet = $('#regSheet'), regForm = $('#regForm'), regName = $('#regName'), regInfo = $('#regInfo'), regErr = $('#regErr');
  let toastT;
  function toast(message) {
    const el = $('#toast');
    el.textContent = message;
    el.classList.add('show');
    clearTimeout(toastT);
    toastT = setTimeout(hideToast, 2600);
  }
  function hideToast() {
    $('#toast').classList.remove('show');
    clearTimeout(toastT);
  }
  function openRegSheet() {
    regErr.hidden = true;
    closeProfileMenu();
    closeSheet();
    hideToast();
    $('#scrim').classList.add('show');
    regSheet.classList.add('show');
    regSheet.setAttribute('aria-hidden', 'false');
    setTimeout(() => regName.focus(), 420);
  }
  function closeRegSheet() {
    regSheet.classList.remove('show');
    regSheet.setAttribute('aria-hidden', 'true');
    if (!$('#sheet').classList.contains('show')) $('#scrim').classList.remove('show');
  }
  $('#closeReg').onclick = closeRegSheet;
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && regSheet.classList.contains('show')) closeRegSheet();
  });
  $('#addPersonTop').onclick = openRegSheet;
  $('#addPersonDemo').onclick = openRegSheet;

  regForm.addEventListener('submit', e => {
    e.preventDefault();
    const name = regName.value.trim();
    if (!name) {
      regErr.hidden = false;
      regName.focus();
      return;
    }
    regErr.hidden = true;
    const info = regInfo.value.trim();
    const id = addProfile(name, info);
    regName.value = ''; regInfo.value = '';
    closeRegSheet();
    selectProfile(id);
    renderOverview();
    toast(`${name} 님을 등록했어요`);
  });

  function prefill() {
    for (let i = 0; i < N; i++) { S.hrH.push(S.baseline.hr + rnd(2.5)); S.edaH.push(S.baseline.eda + rnd(.12)); }
  }

  /* ---------- Boot ---------- */
  render(); renderAlerts(); renderOverview(); setScenario('calm');
  let simulationTimer = setInterval(tickAll, TICK);
  document.addEventListener('visibilitychange', () => {
    clearInterval(simulationTimer);
    if (!document.hidden) simulationTimer = setInterval(tickAll, TICK);
  });
})();
