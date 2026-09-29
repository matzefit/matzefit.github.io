/* Visitor counting for the published viewer. OFF until one ID below is filled in.
 *
 * GitHub Pages reports nothing about a site's visitors, so a counting service is needed.
 * Fill in ONE of the two and republish:
 *
 *   GoatCounter (recommended) -- free for non-commercial use, open source, sets no cookies
 *     (so no consent banner is needed under GDPR). Reports visits, country and region,
 *     browser, operating system, screen size (phone / tablet / desktop) and referrer.
 *     Sign up at https://www.goatcounter.com; the code is the subdomain you choose there
 *     (https://<code>.goatcounter.com is then your dashboard).
 *
 *   Google Analytics 4 -- more detail (city, device model), but it sets cookies, so visitors
 *     from the EU/UK need a consent banner, and more ad blockers suppress it. The ID is the
 *     measurement ID ("G-XXXXXXXXXX") of a GA4 web data stream.
 *
 * Neither identifies individual visitors; both report aggregates, and ad blockers hide a share
 * of visits, so treat the counts as a lower bound. Visits from localhost are never counted.
 *
 * This file is deliberately separate from app.js and loaded on its own: if a blocker refuses
 * it, the viewer still works. app.js reports two events through window.urfTrack, if present:
 * a standpoint being clicked, and the MRT field being switched.
 */
const GOATCOUNTER_CODE = '';   // e.g. 'urf-dubai'  ->  counts go to https://urf-dubai.goatcounter.com
const GA4_ID = '';             // e.g. 'G-ABC123XYZ9'

const LOCAL = ['localhost', '127.0.0.1', '[::1]', ''].includes(location.hostname);

function addScript(src, attrs = {}) {
  const s = document.createElement('script');
  s.async = true;
  s.src = src;
  for (const [k, v] of Object.entries(attrs)) s.setAttribute(k, v);
  document.head.appendChild(s);
}

if (!LOCAL && GOATCOUNTER_CODE) {
  addScript('https://gc.zgo.at/count.js',
    { 'data-goatcounter': `https://${GOATCOUNTER_CODE}.goatcounter.com/count` });
}

if (!LOCAL && GA4_ID) {
  window.dataLayer = window.dataLayer || [];
  window.gtag = function gtag() { window.dataLayer.push(arguments); };   // eslint-disable-line prefer-rest-params
  window.gtag('js', new Date());
  window.gtag('config', GA4_ID);
  addScript(`https://www.googletagmanager.com/gtag/js?id=${encodeURIComponent(GA4_ID)}`);
}

/** An interaction worth counting, e.g. urfTrack('standpoint-click'). A no-op when tracking is off. */
window.urfTrack = (name) => {
  if (LOCAL) return;
  if (GOATCOUNTER_CODE && window.goatcounter?.count) {
    window.goatcounter.count({ path: `event/${name}`, title: name, event: true });
  }
  if (GA4_ID && window.gtag) window.gtag('event', name);
};
