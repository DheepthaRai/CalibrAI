export interface PolicyInputs {
  industry: string;
  costPerViolation: number;
  weeklyVolume: number;
}

export interface PolicyRecommendation {
  recommended_level: number;
  level_name: string;
  level_description: string;
  explanation: string;
  industry_risk_profile: string;
  weekly_fp_cost_estimate: number;
  weekly_attack_cost_estimate: number;
  inputs: PolicyInputs;
}

export interface LevelStats {
  name: string;
  attack_block_rate: number;
  fp_rate: number;
  score: number;
  attack_count: number;
  legit_count: number;
  attacks_blocked: number;
  legit_blocked: number;
}

export interface LevelResult {
  blocked: boolean;
  response: string;
}

export interface QueryResult {
  qid: number;
  query: string;
  is_attack: boolean;
  results: Record<number, LevelResult>;   // level 1–5 → result
}

export interface CalibrationSummary {
  [level: number]: LevelStats;
}

export interface CalibrationState {
  runId: string | null;
  status: 'idle' | 'running' | 'complete' | 'error';
  progress: number;            // 0–100
  completedQueries: number;
  totalQueries: number;
  queries: QueryResult[];
  levelStats: CalibrationSummary;
  recommendedLevel: number | null;
  recommendationText: string;
  error: string | null;
}

export interface AuditRow {
  id: number;
  run_id: string;
  timestamp: string;
  query_text: string;
  is_attack: boolean;
  risk_score: number;
  safety_level: number;
  outcome: 'ALLOWED' | 'BLOCKED';
  cost_justification: string;
  industry: string;
}

export interface OllamaStatus {
  ollama_online: boolean;
  llama_available: boolean;
  deepseek_available: boolean;
  models: string[];
}

export const INDUSTRIES = [
  'Banking',
  'Healthcare',
  'E-commerce',
  'Insurance',
  'Customer Service',
] as const;

export type Industry = (typeof INDUSTRIES)[number];

export const LEVEL_NAMES: Record<number, string> = {
  1: 'Very Strict',
  2: 'Strict',
  3: 'Balanced',
  4: 'Permissive',
  5: 'Very Permissive',
};
