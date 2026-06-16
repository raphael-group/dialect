export const meta = {
  name: 'dialect-lit-validation',
  description: 'Validate DIALECT ME/CO driver-interaction networks against the biological literature across 34 cancer-type groups (find -> adversarially verify -> synthesize)',
  phases: [
    { title: 'Search', detail: 'one literature-search agent per cancer-type group' },
    { title: 'Verify', detail: 'adversarial re-check of established/emerging pairs' },
    { title: 'Synthesize', detail: 'method-recovery + novel-discovery + concordance report' },
  ],
}

// ---- payload injected at generation time (per cancer-type group: cohorts + ME/CO pairs) ----
const GROUPS = /*__PAYLOAD__*/;

// Background DIALECT framing handed to every agent so it judges direction correctly.
const FRAMING = `DIALECT is an EM model that, after subtracting a per-gene/per-sample passenger
background mutation rate (BMR), estimates latent DRIVER mutation status for each gene and then fits a
bivariate-Bernoulli interaction (tau) between gene pairs. It reports:
- ME (mutually exclusive): the two genes' DRIVER mutations co-occur in the same tumor far LESS than
  chance (negative correlation rho<0). Biologically this usually means functional redundancy /
  same-pathway epistasis (one hit suffices) or different molecular subtypes.
- CO (co-occurring): the two genes' driver mutations co-occur MORE than chance (rho>0). Biologically
  this usually means cooperation/synergy, a shared subtype, or a defined genomic context.
Gene symbols may carry _M (missense/in-frame) or _N (truncating/nonsense) effect suffixes; here they
are collapsed to the base gene symbol. BMR support codes: c=CBaSE, d=DIG, m=per-sample MutSig2CV.
A pair supported by 'm' is robust to per-sample tumor-burden confounding (the strongest evidence it is
not a hypermutation artifact); a CO pair seen ONLY under 'c' in a high-burden cohort is the most
suspect.`;

const FINDER_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    group: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          pair: { type: 'string', description: 'GENE_A:GENE_B exactly as given' },
          observed: { type: 'string', enum: ['ME', 'CO'], description: 'direction DIALECT called' },
          status: {
            type: 'string',
            enum: ['established', 'emerging', 'novel', 'contradicted', 'artifact'],
            description: 'established=textbook/multiple studies; emerging=some evidence; novel=biologically plausible but little/no prior report; contradicted=literature shows the OPPOSITE direction; artifact=one or both genes are not credible drivers in this tumor (likely passenger/FLAGS gene)',
          },
          lit_direction: { type: 'string', enum: ['ME', 'CO', 'both', 'none'] },
          concordant: { type: 'boolean', description: 'does DIALECT observed direction match the literature direction?' },
          mechanism: { type: 'string', description: 'one sentence (<=200 chars) on the biology' },
          citations: {
            type: 'array',
            description: 'up to 3 real peer-reviewed sources; empty if novel/artifact',
            items: {
              type: 'object',
              additionalProperties: false,
              properties: {
                ref: { type: 'string', description: 'first-author + journal + year, <=120 chars' },
                identifier: { type: 'string', description: 'PMID, DOI, or URL' },
                year: { type: 'integer' },
              },
              required: ['ref'],
            },
          },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
        required: ['pair', 'observed', 'status', 'lit_direction', 'concordant', 'mechanism', 'confidence'],
      },
    },
  },
  required: ['group', 'findings'],
}

const VERIFIER_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    group: { type: 'string' },
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          pair: { type: 'string' },
          claim_holds: { type: 'boolean', description: 'does the cited evidence really support the claimed status+direction?' },
          final_status: { type: 'string', enum: ['established', 'emerging', 'novel', 'contradicted', 'artifact', 'unverifiable'] },
          final_direction: { type: 'string', enum: ['ME', 'CO', 'both', 'none'] },
          best_citation: { type: 'string', description: 'the single strongest verified citation w/ identifier, or "" if none verifiable' },
          reason: { type: 'string', description: 'why the claim held or failed (<=240 chars)' },
        },
        required: ['pair', 'claim_holds', 'final_status', 'final_direction', 'reason'],
      },
    },
  },
  required: ['group', 'verdicts'],
}

function finderPrompt(g) {
  const cohorts = g.cohorts.map(c => `${c.c} (N=${c.n}, medianTMB=${c.tmb})`).join('; ');
  const fmt = arr => arr.map(p => `${p[0]} [${p[1]},bmr=${p[2]},nCohorts=${p[3]}]`).join('\n  ');
  return `You are a cancer-genomics literature analyst. Validate DIALECT's predicted driver-gene
interaction network for the cancer-type group "${g.group}" against the PEER-REVIEWED biological literature.

${FRAMING}

CONSTITUENT COHORTS: ${cohorts}

DIALECT predicted MUTUALLY-EXCLUSIVE (ME) pairs:
  ${fmt(g.ME) || '(none)'}

DIALECT predicted CO-OCCURRING (CO) pairs:
  ${fmt(g.CO) || '(none)'}

(Tag legend: class dd=both OncoKB cancer genes, dp=one non-OncoKB partner. bmr letters c/d/m as defined above. nCohorts = how many cohorts in this group recovered the pair.)

TASK: For EVERY pair listed, determine what the literature says about that gene pair IN THIS CANCER TYPE
(or closely related context). Use web search to find real evidence — load it first if needed via
ToolSearch query "select:WebSearch,WebFetch", then run multiple targeted WebSearch queries
(e.g. "GENE_A GENE_B mutual exclusivity <cancer>", "GENE_A GENE_B co-occurrence co-mutation <cancer>",
"GENE_A GENE_B <cancer> pathway"). Prefer TCGA marker papers, cBioPortal/MSK studies, COSMIC,
pathway/epistasis papers, and reviews. Fetch a source to confirm when a snippet is ambiguous.

For each pair decide:
- status (established / emerging / novel / contradicted / artifact),
- lit_direction (what direction the literature supports: ME, CO, both, or none),
- concordant (does DIALECT's observed direction match the literature?),
- a one-sentence mechanism, and up to 3 REAL citations with PMIDs/DOIs (NEVER fabricate an identifier;
  if you cannot find a real source, leave citations empty and lower the status to novel or artifact).

Be skeptical and calibrated: only "established" if you can name concrete supporting literature.
If a pair's biology is well known to go the OTHER way than DIALECT called it, mark contradicted
(this is a valuable finding). Return ALL pairs. Your structured output IS the result.`;
}

function verifierPrompt(g, toCheck) {
  const lines = toCheck.map(f =>
    `- ${f.pair} | DIALECT=${f.observed} | claimed status=${f.status}, lit_dir=${f.lit_direction}, concordant=${f.concordant} | mechanism="${f.mechanism}" | cites=${JSON.stringify(f.citations || [])}`
  ).join('\n');
  return `You are an ADVERSARIAL fact-checker for a cancer-genomics rebuttal. Another analyst claimed the
following gene-pair interactions in "${g.group}" are supported by the literature. Independently verify
each one. Default to skepticism: if you cannot confirm a real source supports the claimed STATUS and
DIRECTION, mark claim_holds=false and set final_status to "unverifiable" (or "artifact"/"novel"/"contradicted"
as appropriate).

${FRAMING}

CLAIMS TO CHECK:
${lines}

For EACH claim: load web search if needed (ToolSearch "select:WebSearch,WebFetch"), run your OWN searches,
and (a) confirm the cited identifier (PMID/DOI) actually exists and is about this gene pair in this/related
cancer, and (b) confirm the literature direction (ME vs CO) matches what was claimed. Watch specifically for:
fabricated PMIDs/DOIs, citations that are about a different cancer, and direction errors (claimed CO but the
genes are actually mutually exclusive, or vice versa). Output a verdict per pair with the single strongest
VERIFIED citation (or "" if none). Your structured output IS the result.`;
}

// ---------- Phase 1+2: per-group find -> adversarially verify (pipeline, no barrier) ----------
const perGroup = await pipeline(
  GROUPS,
  g => agent(finderPrompt(g), { label: `find:${g.group}`, phase: 'Search', schema: FINDER_SCHEMA }),
  (found, g) => {
    if (!found) return { group: g.group, found: null, verdicts: [] };
    const toCheck = found.findings.filter(f => f.status === 'established' || f.status === 'emerging' || f.status === 'contradicted');
    if (toCheck.length === 0) return { group: g.group, found, verdicts: [] };
    return agent(verifierPrompt(g, toCheck), { label: `verify:${g.group}`, phase: 'Verify', schema: VERIFIER_SCHEMA })
      .then(v => ({ group: g.group, found, verdicts: (v && v.verdicts) || [] }));
  },
);

// ---------- merge finder + verifier into one master table ----------
const master = [];
for (const r of perGroup.filter(Boolean)) {
  if (!r.found) continue;
  const vmap = {};
  for (const v of r.verdicts) vmap[v.pair] = v;
  for (const f of r.found.findings) {
    const v = vmap[f.pair];
    master.push({
      group: r.group,
      pair: f.pair,
      observed: f.observed,
      status: v ? v.final_status : f.status,
      lit_direction: v ? v.final_direction : f.lit_direction,
      concordant: f.concordant,
      verified: v ? v.claim_holds : (f.status === 'novel' || f.status === 'artifact'),
      confidence: f.confidence,
      mechanism: f.mechanism,
      citation: v && v.best_citation ? v.best_citation
        : (f.citations && f.citations[0] ? `${f.citations[0].ref} ${f.citations[0].identifier || ''}`.trim() : ''),
    });
  }
}
log(`master table: ${master.length} validated pair-findings across ${perGroup.filter(Boolean).length} groups`);

// compact views for the synthesis agents
const established = master.filter(m => (m.status === 'established' || m.status === 'emerging') && m.verified);
const novel = master.filter(m => m.status === 'novel' && (m.confidence === 'high' || m.confidence === 'medium'));
const discordant = master.filter(m => m.status === 'contradicted' || m.concordant === false);
const compact = arr => arr.map(m => `${m.group} | ${m.pair} | DIALECT=${m.observed} | status=${m.status} | litdir=${m.lit_direction} | conf=${m.confidence} | ${m.mechanism} | ${m.citation}`).join('\n');

// ---------- Phase 3: synthesis (3 independent sections, in parallel) ----------
phase('Synthesize');
const STORY_CONTEXT = `This is for a PLOS Comp Biol major-revision rebuttal. The central reviewer critique was that
DIALECT's co-occurrence (CO) calls were inflated by background-mutation-rate (BMR) / hypermutator confounding.
Key findings already established by the authors: (1) a proper per-(gene,sample,context) BMR extracted from a
patched MutSig2CV collapses spurious CO in high-tumor-burden cohorts (e.g. UCEC CO 4850->~300); (2) BUT in
LOW-burden cohorts that same per-sample BMR OVER-corrects and erases REAL biology — e.g. in AML it inflates the
DNMT3A background so much (observed/expected ~1.35) that it deletes the canonical DNMT3A:FLT3 / DNMT3A:IDH1
co-occurrences that CBaSE and DIG both recover; (3) this motivates a burden-aware BMR choice: per-sample MutSig
for high-TMB cohorts, per-gene CBaSE/DIG for low-TMB. The literature validation below tests whether DIALECT's
ME/CO networks recover known cancer biology (method validation) and surface credible novel interactions.`;

const synth = await parallel([
  () => agent(`${STORY_CONTEXT}

You are writing the METHOD-RECOVERY section of the rebuttal's literature-validation appendix. Below are the
gene-pair interactions DIALECT predicted that are SUPPORTED by the literature and survived adversarial
verification. Write a tight, well-organized markdown section that demonstrates DIALECT recovers established
cancer biology. Organize by cancer-type group; for each, give a one-line summary then a bullet list of the
strongest recovered ME and CO pairs with their mechanism and a citation. Lead with the most famous textbook
recoveries (e.g. lung KRAS/EGFR mutual exclusivity, PDAC KRAS:TP53:SMAD4:CDKN2A co-occurrence, glioma
IDH1:TP53:ATRX, AML DNMT3A:FLT3:NPM1, colorectal APC:KRAS:TP53). Note where a CO pair is robust to per-sample
MutSig (bmr code includes m) as the strongest anti-confounding evidence. Be precise and do not invent pairs
not in the list.

VERIFIED ESTABLISHED/EMERGING PAIRS:
${compact(established)}`, { label: 'synth:recovery', phase: 'Synthesize' }),

  () => agent(`${STORY_CONTEXT}

You are writing the NOVEL-CANDIDATES section. Below are biologically plausible gene-pair interactions DIALECT
predicted for which there is little/no prior published report (status=novel) at medium/high confidence. Write a
markdown section highlighting the most interesting candidates as discovery opportunities. Group by cancer type,
prioritize driver-driver pairs and pairs recovered in multiple cohorts or robust to per-sample MutSig. For each,
give the observed direction (ME/CO), a plausible mechanistic hypothesis, and why it is worth experimental or
cohort follow-up. Be honest that these are hypotheses. Do not invent pairs not in the list.

NOVEL CANDIDATE PAIRS:
${compact(novel)}`, { label: 'synth:novel', phase: 'Synthesize' }),

  () => agent(`${STORY_CONTEXT}

You are writing the DIRECTION-CONCORDANCE & CAVEATS section. Below are pairs where DIALECT's ME/CO call is
DISCORDANT with the literature, or was flagged contradicted. Write a markdown section that (1) honestly catalogs
the discordances by cancer type with the likely cause (residual BMR confounding, subtype mixing, or a genuine
novel direction), (2) connects them to the burden-aware BMR story (which discordant CO calls appear only under
CBaSE in high-TMB cohorts vs survive per-sample MutSig), and (3) gives a short, fair assessment of the overall
concordance rate as evidence the method is well-calibrated when the right BMR is used. Do not invent pairs.

DISCORDANT / CONTRADICTED PAIRS:
${compact(discordant)}`, { label: 'synth:concordance', phase: 'Synthesize' }),
]);

return {
  counts: {
    total: master.length,
    established_verified: established.length,
    novel: novel.length,
    discordant: discordant.length,
    groups: perGroup.filter(Boolean).length,
  },
  master,
  sections: {
    recovery: synth[0],
    novel: synth[1],
    concordance: synth[2],
  },
};
