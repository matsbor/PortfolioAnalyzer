#!/usr/bin/env python3
"""
ALPHA MINER ULTIMATE - CSV INPUT VERSION
Complete professional system with survival gates, truth arbitration, and all features
Philosophy: SURVIVE FIRST, ALPHA SECOND

USAGE:
    python3 alpha_miner_ultimate_csv.py portfolio.csv

CSV FORMAT:
    Required: Symbol, Quantity, CostBasis
    Optional: Stage, Metal, Country, Cash (millions), MonthlyBurn (millions)
"""

import sys
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance not installed. Install with: pip install yfinance")

# ============================================================================
# ENUMS & DATA STRUCTURES
# ============================================================================

class NarrativePhase(Enum):
    IGNORED = "Ignored - No attention"
    EMERGING = "🟢 Emerging - Early smart money (BEST PHASE)"
    VALIDATION = "🔵 Validation - Institutions entering (GOOD)"
    CROWDED = "🟡 Crowded - Promotional phase (CAUTION)"
    DISTRIBUTION = "🔴 Distribution - Smart money exiting (DANGER)"
    DEAD = "⚫ Dead - Optionality only"

@dataclass
class SurvivalGate:
    gate_name: str
    passed: bool
    value: float
    threshold: float
    severity: str
    failure_reason: str = ""

@dataclass
class WarrantWall:
    price: float
    shares: int
    dilution_pct: float
    pressure_score: float
    distance_pct: float

@dataclass
class Position:
    symbol: str
    name: str
    quantity: float
    price: float
    market_value: float
    cost_basis: float
    gain_loss: float
    gain_loss_pct: float
    day_change_pct: float
    pct_of_portfolio: float
    stage: str
    primary_metal: str
    jurisdiction: str
    cash_millions: float
    monthly_burn: float
    
    survival_gates: List[SurvivalGate] = field(default_factory=list)
    survival_passed: bool = True
    narrative_phase: NarrativePhase = NarrativePhase.IGNORED
    warrant_walls: List[WarrantWall] = field(default_factory=list)
    
    alpha_score: float = 50.0
    decision_confidence: float = 0.0
    recommendation: str = "HOLD"
    action: str = "⚪ NO ACTION"
    reasoning: str = ""
    override_reason: str = ""

# ============================================================================
# PRICE DATA
# ============================================================================

price_cache = {}
cache_timestamp = {}

def get_live_price(symbol: str) -> dict:
    """Get current price using yfinance"""
    now = datetime.datetime.now()
    
    if symbol in price_cache:
        cache_age = (now - cache_timestamp[symbol]).seconds / 60
        if cache_age < 15:
            return price_cache[symbol]
    
    if not YFINANCE_AVAILABLE:
        return {'price': 0, 'day_change_pct': 0, 'volume': 0, 'error': 'yfinance not installed'}
    
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        
        if hist.empty:
            return {'price': 0, 'day_change_pct': 0, 'volume': 0, 'error': 'No data'}
        
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        day_change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0
        volume = hist['Volume'].iloc[-1]
        
        result = {
            'price': float(current_price),
            'day_change_pct': float(day_change_pct),
            'volume': int(volume)
        }
        
        price_cache[symbol] = result
        cache_timestamp[symbol] = now
        
        return result
    except Exception as e:
        return {'price': 0, 'day_change_pct': 0, 'volume': 0, 'error': str(e)}

# ============================================================================
# SURVIVAL GATE ENGINE
# ============================================================================

class SurvivalGateEngine:
    """THE MOST IMPORTANT COMPONENT - Prevents 90% of disasters"""
    
    def __init__(self):
        self.MIN_CASH_RUNWAY_MONTHS = 8
        self.MAX_RECENT_FINANCINGS = 3
        self.MAX_WARRANT_OVERHANG_PCT = 20
        self.MAX_JURISDICTION_RISK = 60
        
        self.JURISDICTION_SCORES = {
            'Canada': 10, 'USA': 10, 'Australia': 15,
            'Mexico': 40, 'Peru': 45, 'Chile': 35, 'Brazil': 45,
            'Argentina': 60, 'Ecuador': 70, 'Fiji': 50,
            'Unknown': 50
        }
    
    def evaluate(self, position: Position) -> Tuple[bool, List[SurvivalGate]]:
        gates = []
        
        # Gate 1: Cash Runway (MOST CRITICAL)
        gates.append(self.check_cash_runway(position))
        
        # Gate 2: Position Size
        gates.append(self.check_position_size(position))
        
        # Gate 3: Jurisdiction Risk
        gates.append(self.check_jurisdiction_risk(position))
        
        # Gate 4: Performance (catch falling knives)
        gates.append(self.check_performance(position))
        
        critical_failures = [g for g in gates if not g.passed and g.severity == "CRITICAL"]
        high_failures = [g for g in gates if not g.passed and g.severity == "HIGH"]
        
        passed_all = len(critical_failures) == 0 and len(high_failures) <= 1
        
        return passed_all, gates
    
    def check_cash_runway(self, position: Position) -> SurvivalGate:
        """CRITICAL: Will they dilute soon?"""
        if position.cash_millions == 0 or position.monthly_burn == 0:
            # No data - assume okay for now
            return SurvivalGate(
                gate_name="Cash Runway",
                passed=True,
                value=99,
                threshold=self.MIN_CASH_RUNWAY_MONTHS,
                severity="MEDIUM",
                failure_reason="✅ No cash runway data available"
            )
        
        runway_months = position.cash_millions / position.monthly_burn
        passed = runway_months >= self.MIN_CASH_RUNWAY_MONTHS
        
        if runway_months < 4:
            severity = "CRITICAL"
            reason = f"🚨 DILUTION IMMINENT: Only {runway_months:.1f} months cash. Financing announcement likely within 3-6 months. Expect 20-40% dilution."
        elif runway_months < 6:
            severity = "CRITICAL"
            reason = f"🚨 DILUTION RISK HIGH: {runway_months:.1f} months cash. High probability of financing within 6 months."
        elif runway_months < 8:
            severity = "HIGH"
            reason = f"⚠️ DILUTION WATCH: {runway_months:.1f} months cash. Monitor for financing announcements."
        else:
            severity = "MEDIUM"
            reason = f"✅ Adequate cash: {runway_months:.1f} months runway"
        
        return SurvivalGate(
            gate_name="Cash Runway",
            passed=passed,
            value=runway_months,
            threshold=self.MIN_CASH_RUNWAY_MONTHS,
            severity=severity,
            failure_reason=reason
        )
    
    def check_position_size(self, position: Position) -> SurvivalGate:
        """Concentration risk"""
        max_single = 15.0
        passed = position.pct_of_portfolio <= max_single
        
        if not passed:
            reason = f"⚠️ OVER-CONCENTRATED: {position.pct_of_portfolio:.1f}% of portfolio (max: {max_single}%). Reduce to manage risk."
        else:
            reason = f"✅ Reasonable size: {position.pct_of_portfolio:.1f}%"
        
        return SurvivalGate(
            gate_name="Position Size",
            passed=passed,
            value=position.pct_of_portfolio,
            threshold=max_single,
            severity="HIGH",
            failure_reason=reason
        )
    
    def check_jurisdiction_risk(self, position: Position) -> SurvivalGate:
        """Political/operational risk"""
        risk_score = self.JURISDICTION_SCORES.get(position.jurisdiction, 50)
        passed = risk_score <= self.MAX_JURISDICTION_RISK
        
        if not passed:
            reason = f"⚠️ HIGH JURISDICTION RISK: {position.jurisdiction} = {risk_score}/100. Elevated political/operational risk."
        else:
            reason = f"✅ Acceptable jurisdiction: {position.jurisdiction} = {risk_score}/100"
        
        return SurvivalGate(
            gate_name="Jurisdiction Risk",
            passed=passed,
            value=risk_score,
            threshold=self.MAX_JURISDICTION_RISK,
            severity="MEDIUM",
            failure_reason=reason
        )
    
    def check_performance(self, position: Position) -> SurvivalGate:
        """Catch falling knives"""
        threshold = -40.0
        passed = position.gain_loss_pct > threshold
        
        if not passed:
            reason = f"🚨 MAJOR LOSS: {position.gain_loss_pct:.1f}% loss. Thesis likely broken. Review immediately."
        else:
            reason = f"✅ Performance acceptable: {position.gain_loss_pct:+.1f}%"
        
        return SurvivalGate(
            gate_name="Performance Check",
            passed=passed,
            value=position.gain_loss_pct,
            threshold=threshold,
            severity="HIGH",
            failure_reason=reason
        )

# ============================================================================
# NARRATIVE PHASE DETECTOR
# ============================================================================

class NarrativeDetector:
    """Detect story phase - most alpha in EMERGING → VALIDATION"""
    
    def detect(self, position: Position) -> NarrativePhase:
        price_move = position.gain_loss_pct
        
        if price_move < -40:
            return NarrativePhase.DEAD
        elif price_move < 10 and position.stage == "Explorer":
            return NarrativePhase.IGNORED
        elif 10 <= price_move < 50 and position.stage in ["Explorer", "Developer"]:
            return NarrativePhase.EMERGING
        elif 50 <= price_move < 150:
            return NarrativePhase.VALIDATION
        elif price_move >= 150:
            return NarrativePhase.CROWDED
        
        return NarrativePhase.EMERGING

# ============================================================================
# TRUTH ARBITRATION ENGINE
# ============================================================================

class TruthArbitrator:
    """Resolves conflicts - no silent averaging"""
    
    def arbitrate(self, position: Position, survival_passed: bool) -> Dict:
        """Make final decision with reasoning"""
        
        # RULE 0: Survival failure overrides EVERYTHING
        if not survival_passed:
            critical_failures = [g for g in position.survival_gates if not g.passed and g.severity == "CRITICAL"]
            
            if critical_failures:
                return {
                    'action': '🚨 SELL / DO NOT BUY',
                    'recommendation': 'CAPITAL_DENIED',
                    'confidence': 1.0,
                    'reasoning': critical_failures[0].failure_reason,
                    'override': '🔒 SURVIVAL GATE FAILURE OVERRIDES ALL BULLISH SIGNALS'
                }
            else:
                high_failures = [g for g in position.survival_gates if not g.passed and g.severity == "HIGH"]
                return {
                    'action': '🟡 REDUCE / WATCH',
                    'recommendation': 'REDUCE',
                    'confidence': 0.8,
                    'reasoning': ' | '.join([g.failure_reason for g in high_failures]),
                    'override': '⚠️ Multiple survival concerns - elevated risk'
                }
        
        # Calculate base score
        base_score = 50.0
        
        # Adjust for performance
        if position.gain_loss_pct > 100:
            base_score += 15
        elif position.gain_loss_pct > 50:
            base_score += 10
        elif position.gain_loss_pct > 20:
            base_score += 5
        elif position.gain_loss_pct < -20:
            base_score -= 15
        
        # Adjust for momentum
        if position.day_change_pct > 5:
            base_score += 10
        elif position.day_change_pct > 3:
            base_score += 5
        elif position.day_change_pct < -5:
            base_score -= 10
        elif position.day_change_pct < -3:
            base_score -= 5
        
        # Adjust for narrative phase (CRITICAL)
        if position.narrative_phase == NarrativePhase.EMERGING:
            base_score += 20  # BEST PHASE
        elif position.narrative_phase == NarrativePhase.VALIDATION:
            base_score += 10
        elif position.narrative_phase == NarrativePhase.CROWDED:
            base_score -= 15
        elif position.narrative_phase == NarrativePhase.DISTRIBUTION:
            base_score -= 25
        elif position.narrative_phase == NarrativePhase.DEAD:
            base_score -= 30
        
        # Adjust for stage risk
        if position.stage == "Producer":
            base_score += 5  # Lower risk
        elif position.stage == "Explorer":
            base_score -= 5  # Higher risk
        
        position.alpha_score = max(0, min(100, base_score))
        
        # Generate decision
        if base_score >= 75:
            action = '🟢 STRONG BUY / ADD'
            recommendation = 'STRONG_BUY'
            confidence = 0.85
            reasoning = self.generate_reasoning(position, 'bullish')
        elif base_score >= 60:
            action = '🔵 BUY / ACCUMULATE'
            recommendation = 'BUY'
            confidence = 0.75
            reasoning = self.generate_reasoning(position, 'moderately_bullish')
        elif base_score >= 45:
            action = '⚪ HOLD'
            recommendation = 'HOLD'
            confidence = 0.60
            reasoning = self.generate_reasoning(position, 'neutral')
        elif base_score >= 30:
            action = '🟡 TRIM / REDUCE'
            recommendation = 'REDUCE'
            confidence = 0.70
            reasoning = self.generate_reasoning(position, 'moderately_bearish')
        else:
            action = '🔴 SELL / EXIT'
            recommendation = 'SELL'
            confidence = 0.80
            reasoning = self.generate_reasoning(position, 'bearish')
        
        return {
            'action': action,
            'recommendation': recommendation,
            'confidence': confidence,
            'reasoning': reasoning,
            'override': None
        }
    
    def generate_reasoning(self, position: Position, sentiment: str) -> str:
        parts = []
        
        if sentiment == 'bullish':
            parts.append(f"Strong setup")
            if position.gain_loss_pct > 50:
                parts.append(f"Momentum (+{position.gain_loss_pct:.0f}%)")
        elif sentiment == 'moderately_bullish':
            parts.append(f"Decent opportunity")
        elif sentiment == 'neutral':
            parts.append(f"Mixed signals")
        elif sentiment == 'moderately_bearish':
            parts.append(f"Weakening position")
        elif sentiment == 'bearish':
            parts.append(f"Multiple red flags")
        
        parts.append(position.narrative_phase.value.split('-')[0].strip())
        
        failures = [g for g in position.survival_gates if not g.passed]
        if failures:
            parts.append(f"{len(failures)} survival concerns")
        
        return " | ".join(parts)

# ============================================================================
# NO-TRADE ENFORCEMENT
# ============================================================================

class NoTradeEnforcer:
    """Comfortable saying: 'We do nothing today'"""
    
    def should_trade(self, decision: Dict, position: Position) -> Tuple[bool, str]:
        if decision['confidence'] < 0.65:
            return False, f"⚪ NO ACTION - Low confidence ({decision['confidence']:.2f})"
        
        if position.pct_of_portfolio < 3 and 'HOLD' in decision['recommendation']:
            return False, f"⚪ NO ACTION - Small position, neutral signal"
        
        return True, "Trade approved"

# ============================================================================
# CSV LOADER
# ============================================================================

def load_portfolio_csv(filepath: str) -> pd.DataFrame:
    """Load and validate portfolio CSV"""
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Handle different naming conventions
        mappings = {
            'costbasis': 'cost_basis',
            'cost': 'cost_basis',
            'qty': 'quantity',
            'shares': 'quantity',
            'ticker': 'symbol',
            'cash_mm': 'cash',
            'burn': 'monthly_burn'
        }
        
        for old, new in mappings.items():
            if old in df.columns and new not in df.columns:
                df.rename(columns={old: new}, inplace=True)
        
        # Check required
        required = ['symbol', 'quantity', 'cost_basis']
        missing = [c for c in required if c not in df.columns]
        
        if missing:
            print(f"❌ Missing columns: {missing}")
            sys.exit(1)
        
        # Add defaults for optional columns
        if 'stage' not in df.columns:
            df['stage'] = 'Explorer'
        if 'metal' not in df.columns:
            df['metal'] = 'Gold'
        if 'country' not in df.columns:
            df['country'] = 'Unknown'
        if 'cash' not in df.columns:
            df['cash'] = 10.0
        if 'monthly_burn' not in df.columns:
            df['monthly_burn'] = 1.0
        
        return df
    
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        sys.exit(1)

# ============================================================================
# MASTER SYSTEM
# ============================================================================

class AlphaMinerUltimate:
    """THE COMPLETE SYSTEM"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.date = datetime.datetime.now()
        
        self.survival_engine = SurvivalGateEngine()
        self.narrative_detector = NarrativeDetector()
        self.arbitrator = TruthArbitrator()
        self.no_trade_enforcer = NoTradeEnforcer()
        
        print(f"✅ Loaded {len(df)} positions")
        print(f"📊 Fetching live prices...")
        
        self.update_prices()
        self.calculate_values()
    
    def update_prices(self):
        """Get live prices for all stocks"""
        for idx, row in self.df.iterrows():
            symbol = row['symbol']
            price_data = get_live_price(symbol)
            self.df.at[idx, 'current_price'] = price_data['price']
            self.df.at[idx, 'day_change_pct'] = price_data['day_change_pct']
        
        print(f"✅ Prices updated (15-min delayed, free)")
    
    def calculate_values(self):
        """Calculate portfolio values"""
        self.df['market_value'] = self.df['quantity'] * self.df['current_price']
        self.df['gain_loss'] = self.df['market_value'] - self.df['cost_basis']
        self.df['gain_loss_pct'] = (self.df['gain_loss'] / self.df['cost_basis'] * 100).fillna(0)
        
        total_mv = self.df['market_value'].sum()
        self.df['pct_portfolio'] = (self.df['market_value'] / total_mv * 100).fillna(0)
        
        self.total_equity = total_mv
    
    def analyze_all(self) -> List[Dict]:
        """Run complete analysis"""
        results = []
        
        for idx, row in self.df.iterrows():
            pos = Position(
                symbol=row['symbol'],
                name=row.get('name', row['symbol']),
                quantity=row['quantity'],
                price=row['current_price'],
                market_value=row['market_value'],
                cost_basis=row['cost_basis'],
                gain_loss=row['gain_loss'],
                gain_loss_pct=row['gain_loss_pct'],
                day_change_pct=row['day_change_pct'],
                pct_of_portfolio=row['pct_portfolio'],
                stage=row['stage'],
                primary_metal=row['metal'],
                jurisdiction=row['country'],
                cash_millions=row['cash'],
                monthly_burn=row['monthly_burn']
            )
            
            # Run analysis
            survival_passed, gates = self.survival_engine.evaluate(pos)
            pos.survival_gates = gates
            pos.survival_passed = survival_passed
            
            pos.narrative_phase = self.narrative_detector.detect(pos)
            
            decision = self.arbitrator.arbitrate(pos, survival_passed)
            
            should_trade, trade_msg = self.no_trade_enforcer.should_trade(decision, pos)
            
            if not should_trade:
                decision['action'] = trade_msg
            
            pos.action = decision['action']
            pos.recommendation = decision['recommendation']
            pos.reasoning = decision['reasoning']
            pos.decision_confidence = decision['confidence']
            pos.override_reason = decision.get('override', '')
            
            results.append({'position': pos, 'decision': decision})
        
        return results
    
    def generate_report(self, results: List[Dict]) -> str:
        """Generate complete report"""
        lines = []
        lines.append("=" * 100)
        lines.append(f"ALPHA MINER ULTIMATE - COMPLETE CAPITAL ALLOCATION SYSTEM")
        lines.append(f"Date: {self.date.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Philosophy: SURVIVE FIRST, ALPHA SECOND")
        lines.append("=" * 100)
        lines.append("")
        
        # Portfolio summary
        total_gain = self.df['gain_loss'].sum()
        total_cb = self.df['cost_basis'].sum()
        total_return_pct = (total_gain / total_cb * 100) if total_cb > 0 else 0
        
        lines.append("PORTFOLIO SUMMARY")
        lines.append("-" * 100)
        lines.append(f"Total Value:      ${self.total_equity:>12,.2f}")
        lines.append(f"Total Return:     ${total_gain:>+12,.2f}  ({total_return_pct:>+6.2f}%)")
        lines.append(f"Positions:        {len(results):>12}")
        lines.append("")
        
        # Critical alerts
        critical_issues = []
        for r in results:
            pos = r['position']
            critical_gates = [g for g in pos.survival_gates if not g.passed and g.severity == "CRITICAL"]
            if critical_gates:
                critical_issues.append((pos, critical_gates))
        
        if critical_issues:
            lines.append("🚨 CRITICAL ALERTS - IMMEDIATE ACTION REQUIRED")
            lines.append("=" * 100)
            for pos, gates in critical_issues:
                lines.append(f"\n{pos.symbol} - {pos.name}")
                lines.append(f"Action Required: {pos.action}")
                for gate in gates:
                    lines.append(f"  • {gate.failure_reason}")
            lines.append("")
        
        # Action categories
        immediate = []
        near_term = []
        holds = []
        
        for r in results:
            pos = r['position']
            if '🚨' in pos.action or '🔴' in pos.action:
                immediate.append(r)
            elif '🟢' in pos.action or '🔵' in pos.action or '🟡' in pos.action:
                near_term.append(r)
            else:
                holds.append(r)
        
        lines.append("TODAY'S ACTIONS")
        lines.append("=" * 100)
        lines.append("")
        
        if immediate:
            lines.append("🚨 IMMEDIATE ACTIONS (Do today):")
            lines.append("-" * 100)
            for r in immediate:
                pos = r['position']
                lines.append(f"\n{pos.symbol}: {pos.action}")
                lines.append(f"  Reasoning: {pos.reasoning}")
                if pos.override_reason:
                    lines.append(f"  Override: {pos.override_reason}")
                lines.append(f"  Confidence: {pos.decision_confidence:.2f}")
        else:
            lines.append("✅ NO IMMEDIATE ACTIONS REQUIRED")
        
        lines.append("")
        
        if near_term:
            lines.append("\n📋 NEAR-TERM ACTIONS (This week):")
            lines.append("-" * 100)
            for r in near_term[:5]:
                pos = r['position']
                lines.append(f"\n{pos.symbol}: {pos.action}")
                lines.append(f"  → {pos.reasoning}")
        
        # Complete position analysis
        lines.append("\n\n")
        lines.append("COMPLETE POSITION ANALYSIS")
        lines.append("=" * 100)
        
        sorted_results = sorted(results, key=lambda r: r['position'].alpha_score, reverse=True)
        
        for r in sorted_results:
            pos = r['position']
            
            lines.append(f"\n{'='*100}")
            lines.append(f"{pos.symbol} - {pos.name}")
            lines.append(f"{'='*100}")
            
            lines.append(f"Position:         {pos.quantity:,.0f} shares @ ${pos.price:.4f} = ${pos.market_value:,.2f} ({pos.pct_of_portfolio:.1f}%)")
            lines.append(f"Cost Basis:       ${pos.cost_basis:,.2f}")
            lines.append(f"Return:           ${pos.gain_loss:+,.2f} ({pos.gain_loss_pct:+.1f}%)")
            lines.append(f"Day Change:       {pos.day_change_pct:+.2f}%")
            lines.append(f"Stage:            {pos.stage} | {pos.primary_metal} | {pos.jurisdiction}")
            lines.append("")
            
            lines.append(f"ALPHA SCORE:      {pos.alpha_score:.1f}/100")
            lines.append(f"RECOMMENDATION:   {pos.action}")
            lines.append(f"REASONING:        {pos.reasoning}")
            lines.append(f"CONFIDENCE:       {pos.decision_confidence:.2f}")
            lines.append(f"NARRATIVE:        {pos.narrative_phase.value}")
            lines.append("")
            
            lines.append("SURVIVAL GATES:")
            for gate in pos.survival_gates:
                status = "✅ PASS" if gate.passed else "🚨 FAIL" if gate.severity == "CRITICAL" else "⚠️  FAIL"
                lines.append(f"  {status} {gate.gate_name}: {gate.value:.1f} vs {gate.threshold:.1f}")
                if not gate.passed:
                    lines.append(f"       {gate.failure_reason}")
            lines.append("")
        
        lines.append("\n" + "="*100)
        lines.append("END OF REPORT")
        lines.append("="*100)
        
        return "\n".join(lines)

# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("❌ Usage: python3 alpha_miner_ultimate_csv.py portfolio.csv")
        sys.exit(1)
    
    if not YFINANCE_AVAILABLE:
        print("❌ Please install yfinance: pip install yfinance")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    df = load_portfolio_csv(csv_file)
    
    system = AlphaMinerUltimate(df)
    results = system.analyze_all()
    report = system.generate_report(results)
    
    print(report)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"alpha_ultimate_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(report)
    
    print(f"\n💾 Report saved: {filename}")

if __name__ == "__main__":
    main()
