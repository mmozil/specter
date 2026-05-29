"""Tools do agente Murdock — cada tool é chamável pelo Pydantic AI agent."""
import logging
from dataclasses import dataclass
from typing import Optional

from pydantic_ai import RunContext
from sqlalchemy.ext.asyncio import AsyncSession

from src.rag.search import hybrid_search

logger = logging.getLogger(__name__)


# ── Deps compartilhadas ───────────────────────────────────────────────────

@dataclass
class MurdockDeps:
    """Dependências injetadas em todas as tools."""
    db: AsyncSession


# ═══════════════════════════════════════════════════════════════════════════
# Tool 1: Busca de Legislação
# ═══════════════════════════════════════════════════════════════════════════

async def search_law(
    ctx: RunContext[MurdockDeps],
    query: str,
    source_type: Optional[str] = None,
) -> str:
    """Busca legislação brasileira na knowledge base de fontes oficiais gov.br.

    Use para consultar leis, decretos, instruções normativas, resoluções e qualquer
    texto legal. Fontes: Planalto, Receita Federal, CONFAZ, CGSN.

    Args:
        query: Texto da consulta (ex: "alíquota ICMS interestadual", "fator R Simples Nacional")
        source_type: Filtro opcional — fiscal_legislacao, fiscal_ncm, fiscal_confaz, fiscal_reforma, fiscal_simples, fiscal_stf, fiscal_stj
    """
    results = await hybrid_search(ctx.deps.db, query, limit=6, source_type=source_type)

    if not results:
        return "Nenhum resultado encontrado na knowledge base para essa consulta."

    parts = []
    for i, r in enumerate(results, 1):
        src = r.get("source", {})
        section = r.get("section", "")
        section_str = f" ({section})" if section else ""
        parts.append(
            f"[{i}] {src.get('title', 'Sem título')}{section_str}\n"
            f"    Fundamentação: {src.get('fundamentacao', 'N/A')}\n"
            f"    Similaridade: {r.get('rrf_score', 0):.4f}\n"
            f"    Fonte: {src.get('url', 'N/A')}\n"
            f"    Conteúdo:\n    {r.get('content', '')[:800]}"
        )

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Tool 2: Cálculo Tributário
# ═══════════════════════════════════════════════════════════════════════════

# Tabelas Simples Nacional 2026 (LC 123/2006)
_SIMPLES_ANEXOS = {
    "anexo_i": {  # Comércio
        "nome": "Anexo I — Comércio",
        "faixas": [
            (180_000, 0.04, 0),
            (360_000, 0.073, 5_940),
            (720_000, 0.095, 13_860),
            (1_800_000, 0.107, 22_500),
            (3_600_000, 0.143, 87_300),
            (4_800_000, 0.19, 378_000),
        ],
    },
    "anexo_ii": {  # Indústria
        "nome": "Anexo II — Indústria",
        "faixas": [
            (180_000, 0.045, 0),
            (360_000, 0.078, 5_940),
            (720_000, 0.10, 13_860),
            (1_800_000, 0.112, 22_500),
            (3_600_000, 0.147, 85_500),
            (4_800_000, 0.30, 720_000),
        ],
    },
    "anexo_iii": {  # Serviços (fator R >= 28%)
        "nome": "Anexo III — Serviços",
        "faixas": [
            (180_000, 0.06, 0),
            (360_000, 0.112, 9_360),
            (720_000, 0.135, 17_640),
            (1_800_000, 0.16, 35_640),
            (3_600_000, 0.21, 125_640),
            (4_800_000, 0.33, 648_000),
        ],
    },
    "anexo_iv": {  # Serviços cessão mão-de-obra
        "nome": "Anexo IV — Serviços (cessão mão-de-obra)",
        "faixas": [
            (180_000, 0.045, 0),
            (360_000, 0.09, 8_100),
            (720_000, 0.102, 12_420),
            (1_800_000, 0.14, 39_780),
            (3_600_000, 0.22, 183_780),
            (4_800_000, 0.33, 828_000),
        ],
    },
    "anexo_v": {  # Serviços intelectuais (fator R < 28%)
        "nome": "Anexo V — Serviços intelectuais",
        "faixas": [
            (180_000, 0.155, 0),
            (360_000, 0.18, 4_500),
            (720_000, 0.195, 9_900),
            (1_800_000, 0.205, 17_100),
            (3_600_000, 0.23, 62_100),
            (4_800_000, 0.305, 540_000),
        ],
    },
}

# Presunção Lucro Presumido (IRPJ / CSLL)
_LP_PRESUNCAO = {
    "comercio": (0.08, 0.12),
    "industria": (0.08, 0.12),
    "servicos": (0.32, 0.32),
    "transporte_carga": (0.08, 0.12),
    "transporte_passageiros": (0.16, 0.12),
    "servicos_hospitalares": (0.08, 0.12),
    "revenda_combustiveis": (0.016, 0.12),
}


def _calc_simples(rbt12: float, anexo: str, fator_r: float = 0.0, receita_mes: float = 0.0) -> dict:
    """Calcula alíquota efetiva do Simples Nacional.

    `receita_mes`: receita do mês corrente (DAS = receita_mes × alíquota efetiva).
    Se não informada, usa rbt12/12 como proxy.
    """
    # Fator R: se >= 28% e atividade Anexo V, migra para III
    if anexo == "anexo_v" and fator_r >= 0.28:
        anexo = "anexo_iii"

    tabela = _SIMPLES_ANEXOS.get(anexo)
    if not tabela:
        return {"erro": f"Anexo '{anexo}' não encontrado"}

    if rbt12 > 4_800_000:
        return {"erro": "Receita bruta ultrapassa o limite do Simples Nacional (R$4.800.000)"}

    base_mes = receita_mes if receita_mes else (rbt12 / 12)
    for limite, aliq_nominal, parcela_deduzir in tabela["faixas"]:
        if rbt12 <= limite:
            aliq_efetiva = (rbt12 * aliq_nominal - parcela_deduzir) / rbt12
            das_mes = base_mes * aliq_efetiva
            return {
                "regime": "Simples Nacional",
                "anexo": tabela["nome"],
                "fator_r": f"{fator_r:.1%}" if fator_r else "N/A",
                "rbt12": f"R${rbt12:,.2f}",
                "faixa_limite": f"R${limite:,.2f}",
                "aliquota_nominal": f"{aliq_nominal:.2%}",
                "parcela_deduzir": f"R${parcela_deduzir:,.2f}",
                "aliquota_efetiva": f"{aliq_efetiva:.4%}",
                "receita_do_mes": f"R${base_mes:,.2f}",
                "das_do_mes": f"R${das_mes:,.2f}",
                "imposto_anual_estimado": f"R${(rbt12 * aliq_efetiva):,.2f}",
            }

    return {"erro": "Faixa não encontrada"}


def _calc_lucro_presumido(receita_mensal: float, atividade: str) -> dict:
    """Calcula carga tributária no Lucro Presumido."""
    presuncao = _LP_PRESUNCAO.get(atividade, (0.32, 0.32))
    pres_irpj, pres_csll = presuncao

    base_irpj = receita_mensal * pres_irpj
    base_csll = receita_mensal * pres_csll

    irpj = base_irpj * 0.15
    adicional_irpj = max(0, base_irpj - 20_000) * 0.10
    csll = base_csll * 0.09
    pis = receita_mensal * 0.0065
    cofins = receita_mensal * 0.03

    total = irpj + adicional_irpj + csll + pis + cofins
    aliq_efetiva = total / receita_mensal if receita_mensal else 0

    return {
        "regime": "Lucro Presumido",
        "atividade": atividade,
        "receita_mensal": f"R${receita_mensal:,.2f}",
        "presuncao_irpj": f"{pres_irpj:.0%}",
        "presuncao_csll": f"{pres_csll:.0%}",
        "base_irpj": f"R${base_irpj:,.2f}",
        "base_csll": f"R${base_csll:,.2f}",
        "irpj_15pct": f"R${irpj:,.2f}",
        "adicional_irpj_10pct": f"R${adicional_irpj:,.2f}",
        "csll_9pct": f"R${csll:,.2f}",
        "pis_065pct": f"R${pis:,.2f}",
        "cofins_3pct": f"R${cofins:,.2f}",
        "total_mensal": f"R${total:,.2f}",
        "total_anual": f"R${total * 12:,.2f}",
        "aliquota_efetiva": f"{aliq_efetiva:.4%}",
        "nota": "IN RFB 2.306/2026: presunções majoradas em 20% para receita >R$5M/ano",
    }


def _calc_lucro_real(receita_mensal: float, lucro_mensal: float) -> dict:
    """Calcula carga tributária no Lucro Real (simplificado)."""
    irpj = lucro_mensal * 0.15
    adicional_irpj = max(0, lucro_mensal - 20_000) * 0.10
    csll = lucro_mensal * 0.09
    pis = receita_mensal * 0.0165  # Não-cumulativo (créditos não considerados aqui)
    cofins = receita_mensal * 0.076

    total = irpj + adicional_irpj + csll + pis + cofins
    aliq_efetiva = total / receita_mensal if receita_mensal else 0

    return {
        "regime": "Lucro Real",
        "receita_mensal": f"R${receita_mensal:,.2f}",
        "lucro_mensal": f"R${lucro_mensal:,.2f}",
        "margem_lucro": f"{lucro_mensal / receita_mensal:.1%}" if receita_mensal else "N/A",
        "irpj_15pct": f"R${irpj:,.2f}",
        "adicional_irpj_10pct": f"R${adicional_irpj:,.2f}",
        "csll_9pct": f"R${csll:,.2f}",
        "pis_165pct_nao_cumulativo": f"R${pis:,.2f}",
        "cofins_76pct_nao_cumulativo": f"R${cofins:,.2f}",
        "total_mensal_bruto": f"R${total:,.2f}",
        "aliquota_efetiva_bruta": f"{aliq_efetiva:.4%}",
        "nota": "PIS/COFINS não-cumulativo: créditos sobre insumos reduzem valor efetivo. Calcule créditos separadamente.",
    }


async def calculate_tax(
    ctx: RunContext[MurdockDeps],
    regime: str,
    receita_mensal: float,
    atividade: str = "servicos",
    anexo_simples: str = "anexo_iii",
    folha_salarios: float = 0.0,
    lucro_mensal: float = 0.0,
    rbt12: float = 0.0,
) -> str:
    """Calcula carga tributária estimada para um regime tributário brasileiro.

    Use para comparar regimes, simular cenários e calcular impostos.

    Args:
        regime: "simples", "lucro_presumido" ou "lucro_real"
        receita_mensal: Receita bruta do MÊS corrente em R$ (base do DAS no Simples)
        atividade: "comercio", "industria", "servicos", "transporte_carga", "transporte_passageiros", "servicos_hospitalares", "revenda_combustiveis"
        anexo_simples: Anexo do Simples — "anexo_iii" a "anexo_v" para serviços. Para comércio/indústria é DERIVADO da atividade automaticamente (ignore).
        folha_salarios: Folha de salários mensal em R$ (para cálculo do fator R no Simples)
        lucro_mensal: Lucro mensal estimado em R$ (para Lucro Real)
        rbt12: Receita bruta dos últimos 12 meses em R$ (define a faixa/alíquota). Se 0, assume receita_mensal × 12.
    """
    rbt12_calc = rbt12 if rbt12 else receita_mensal * 12

    if regime == "simples":
        # Anexo DERIVADO da atividade — não confiar na classificação do modelo
        # (comércio = Anexo I, indústria = Anexo II; serviços usa o anexo informado)
        if atividade == "comercio":
            anexo_simples = "anexo_i"
        elif atividade == "industria":
            anexo_simples = "anexo_ii"
        fator_r = (folha_salarios / receita_mensal) if receita_mensal and folha_salarios else 0.0
        result = _calc_simples(rbt12_calc, anexo_simples, fator_r, receita_mes=receita_mensal)
    elif regime == "lucro_presumido":
        result = _calc_lucro_presumido(receita_mensal, atividade)
    elif regime == "lucro_real":
        lucro = lucro_mensal if lucro_mensal else receita_mensal * 0.15
        result = _calc_lucro_real(receita_mensal, lucro)
    else:
        return f"Regime '{regime}' não reconhecido. Use: simples, lucro_presumido ou lucro_real."

    lines = [f"═══ {result.get('regime', regime).upper()} ═══"]
    for k, v in result.items():
        if k == "regime":
            continue
        label = k.replace("_", " ").title()
        lines.append(f"  {label}: {v}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Tool 3: Consulta NCM
# ═══════════════════════════════════════════════════════════════════════════

async def check_ncm(
    ctx: RunContext[MurdockDeps],
    query: str,
) -> str:
    """Consulta NCM (Nomenclatura Comum do Mercosul) na knowledge base.

    Busca por código NCM ou descrição de produto para encontrar classificação fiscal,
    alíquota de IPI, e informações de substituição tributária.

    Args:
        query: Código NCM (ex: "8471.30.19") ou descrição do produto (ex: "notebook portátil")
    """
    results = await hybrid_search(
        ctx.deps.db, f"NCM {query}", limit=5, source_type="fiscal_ncm"
    )

    if not results:
        # Fallback: busca geral
        results = await hybrid_search(ctx.deps.db, f"classificação fiscal {query}", limit=5)

    if not results:
        return (
            f"NCM '{query}' não encontrado na knowledge base. "
            "Recomendo consultar o SISCOMEX: https://portalunico.siscomex.gov.br/classif/"
        )

    parts = [f"Resultados para NCM '{query}':"]
    for i, r in enumerate(results, 1):
        parts.append(
            f"\n[{i}] {r.get('source', {}).get('title', 'N/A')}\n"
            f"    {r.get('content', '')[:600]}"
        )
    parts.append(
        "\nFonte oficial: SISCOMEX — https://portalunico.siscomex.gov.br/classif/"
    )

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Tool 4: Reforma Tributária 2026
# ═══════════════════════════════════════════════════════════════════════════

_CRONOGRAMA_REFORMA = """
CRONOGRAMA DA REFORMA TRIBUTÁRIA DO CONSUMO (EC 132/2023 + LC 214/2025):

2026 — Fase de teste
  • CBS (federal): 0,9% (teste)
  • IBS (estadual/municipal): 0,1% (teste)
  • PIS/Cofins, ICMS e ISS mantidos integralmente
  • Crédito integral de CBS/IBS pagos na fase teste

2027 — CBS plena
  • CBS substitui PIS/Cofins integralmente
  • IBS continua em 0,1% (teste)
  • ICMS e ISS mantidos

2029-2032 — Transição gradual ICMS/ISS → IBS
  • 2029: ICMS/ISS reduzem 10% (IBS sobe proporcionalmente)
  • 2030: ICMS/ISS reduzem 25%
  • 2031: ICMS/ISS reduzem 50%
  • 2032: ICMS/ISS reduzem 75%

2033 — IBS pleno
  • ICMS e ISS extintos
  • IBS (estadual+municipal) em alíquota plena
  • Alíquota de referência estimada: ~26,5% (CBS + IBS)

SPLIT PAYMENT — Obrigatório a partir de 2026:
  • Recolhimento automático na liquidação financeira
  • Banco/adquirente separa tributo no pagamento
  • Reduz inadimplência e sonegação

REGIMES ESPECÍFICOS:
  • Combustíveis, financeiro, imobiliário, cooperativas, nano/microempresas
  • Simples Nacional: opção de recolher CBS/IBS por fora (para dar crédito ao comprador)
"""


async def reform_2026(
    ctx: RunContext[MurdockDeps],
    query: str,
) -> str:
    """Consulta sobre a Reforma Tributária do Consumo (CBS/IBS).

    Inclui cronograma de transição 2026-2033, split payment, regimes específicos,
    e impacto nos negócios. Fontes: EC 132/2023, LC 214/2025, Ministério da Fazenda.

    Args:
        query: Pergunta sobre a reforma (ex: "quando ICMS será extinto?", "split payment como funciona?")
    """
    # Busca na knowledge base
    results = await hybrid_search(
        ctx.deps.db, query, limit=5, source_type="fiscal_reforma"
    )

    # Se não encontrou reforma, busca legislação geral
    if not results:
        results = await hybrid_search(ctx.deps.db, f"reforma tributária {query}", limit=5)

    parts = [_CRONOGRAMA_REFORMA]

    if results:
        parts.append("\n═══ FONTES OFICIAIS ═══")
        for i, r in enumerate(results, 1):
            src = r.get("source", {})
            parts.append(
                f"\n[{i}] {src.get('title', 'N/A')}\n"
                f"    Fundamentação: {src.get('fundamentacao', 'N/A')}\n"
                f"    {r.get('content', '')[:600]}"
            )

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Tool 5: Recuperação de Créditos Tributários
# ═══════════════════════════════════════════════════════════════════════════

async def credit_recovery(
    ctx: RunContext[MurdockDeps],
    query: str,
    regime: str = "lucro_real",
) -> str:
    """Analisa oportunidades de recuperação de créditos tributários.

    Identifica créditos de PIS/COFINS, ICMS, IPI e outros tributos passíveis
    de recuperação administrativa ou judicial.

    Args:
        query: Descrição da situação (ex: "créditos PIS/COFINS sobre energia elétrica", "ICMS-ST pago a maior")
        regime: Regime tributário — "simples", "lucro_presumido", "lucro_real"
    """
    results = await hybrid_search(
        ctx.deps.db,
        f"crédito tributário recuperação {query}",
        limit=6,
    )

    parts = [
        "═══ ANÁLISE DE CRÉDITOS TRIBUTÁRIOS ═══",
        f"Regime: {regime.replace('_', ' ').title()}",
    ]

    if regime == "simples":
        parts.append(
            "\n⚠ ATENÇÃO: Empresas do Simples Nacional têm direito LIMITADO a créditos. "
            "Créditos de PIS/COFINS e ICMS não se aplicam na maioria dos casos. "
            "Exceções: ICMS-ST pago a maior (restituição), PIS/COFINS monofásico recolhido indevidamente."
        )
    elif regime == "lucro_presumido":
        parts.append(
            "\n⚠ ATENÇÃO: No Lucro Presumido, PIS/COFINS são cumulativos (sem direito a créditos de insumos). "
            "Oportunidades: ICMS-ST restituição, PIS/COFINS monofásico, exclusão ICMS da base PIS/COFINS (Tema 69 STF)."
        )

    if results:
        parts.append("\n═══ LEGISLAÇÃO E JURISPRUDÊNCIA ═══")
        for i, r in enumerate(results, 1):
            src = r.get("source", {})
            parts.append(
                f"\n[{i}] {src.get('title', 'N/A')} — {src.get('fundamentacao', 'N/A')}\n"
                f"    {r.get('content', '')[:500]}"
            )
    else:
        parts.append(
            "\nConhecimento base não retornou resultados específicos. "
            "Recomendo consultar: Lei 10.637/2002 (PIS), Lei 10.833/2003 (COFINS), "
            "LC 87/1996 (ICMS), e jurisprudência STJ/STF."
        )

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Tool 6: Calendário Fiscal
# ═══════════════════════════════════════════════════════════════════════════

_CALENDARIO_FISCAL = {
    "simples": [
        ("Dia 20", "DAS — Documento de Arrecadação do Simples Nacional"),
        ("Dia 20", "FGTS Digital — depósito mensal"),
        ("Dia 15", "INSS empregado doméstico (DAE)"),
        ("Último dia útil", "DeSTDA — Declaração de ST, DIFAL e antecipação (ICMS)"),
        ("31/01", "DEFIS — Declaração anual do Simples Nacional"),
        ("31/05", "DIRPF — Declaração de Imposto de Renda PF (sócios)"),
    ],
    "lucro_presumido": [
        ("Dia 20", "DARF — IRPJ (trimestral: jan/abr/jul/out)"),
        ("Dia 20", "DARF — CSLL (trimestral: jan/abr/jul/out)"),
        ("Dia 25", "DARF — PIS (mensal)"),
        ("Dia 25", "DARF — COFINS (mensal)"),
        ("Dia 20", "FGTS Digital — depósito mensal"),
        ("Dia 20", "DCTFWeb/MIT — transmissão mensal"),
        ("Dia 15", "EFD-Contribuições — mensal"),
        ("Último dia útil jul", "ECD — Escrituração Contábil Digital (anual)"),
        ("Último dia útil jul", "ECF — Escrituração Contábil Fiscal (anual)"),
        ("31/05", "DIRPF (sócios)"),
    ],
    "lucro_real": [
        ("Dia 20/último dia útil", "DARF — IRPJ (estimativa mensal ou trimestral)"),
        ("Dia 20/último dia útil", "DARF — CSLL (estimativa mensal ou trimestral)"),
        ("Dia 25", "DARF — PIS não-cumulativo (mensal)"),
        ("Dia 25", "DARF — COFINS não-cumulativo (mensal)"),
        ("Dia 20", "FGTS Digital — depósito mensal"),
        ("Dia 20", "DCTFWeb/MIT — transmissão mensal"),
        ("Dia 15", "EFD-Contribuições — mensal"),
        ("Dia 25", "EFD ICMS/IPI — mensal (se contribuinte ICMS)"),
        ("Último dia útil jul", "ECD (anual)"),
        ("Último dia útil jul", "ECF (anual)"),
        ("Último dia útil jan", "DIRF — Declaração do IR Retido na Fonte"),
        ("31/05", "DIRPF (sócios)"),
    ],
}


async def calendar(
    ctx: RunContext[MurdockDeps],
    regime: str = "simples",
    mes: int = 0,
) -> str:
    """Retorna o calendário de obrigações fiscais e prazos tributários.

    Lista todas as obrigações mensais e anuais por regime tributário.

    Args:
        regime: "simples", "lucro_presumido" ou "lucro_real"
        mes: Mês específico (1-12) ou 0 para todas as obrigações
    """
    regime_key = regime.lower().replace(" ", "_")
    obrigacoes = _CALENDARIO_FISCAL.get(regime_key)

    if not obrigacoes:
        return f"Regime '{regime}' não reconhecido. Use: simples, lucro_presumido, lucro_real."

    # Busca complementar na knowledge base
    results = await hybrid_search(
        ctx.deps.db,
        f"obrigações acessórias prazos {regime}",
        limit=3,
    )

    parts = [f"═══ CALENDÁRIO FISCAL — {regime.upper().replace('_', ' ')} ═══\n"]

    for prazo, descricao in obrigacoes:
        parts.append(f"  📅 {prazo}: {descricao}")

    parts.append("\n═══ PENALIDADES POR ATRASO ═══")
    parts.append("  • DAS/DARF: multa 0,33%/dia (máx 20%) + SELIC")
    parts.append("  • DCTF/DCTFWeb: multa mínima R$500 (LP/LR) ou R$200 (Simples/inativas)")
    parts.append("  • ECD/ECF: multa 0,25% do lucro líquido (mínimo R$500)")
    parts.append("  • EFD-Contribuições: R$500/mês (LP) ou R$1.500/mês (LR)")

    if results:
        parts.append("\n═══ INFORMAÇÕES COMPLEMENTARES ═══")
        for r in results[:2]:
            parts.append(f"  {r.get('content', '')[:300]}")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Tool 7: Jurisprudência STF/STJ
# ═══════════════════════════════════════════════════════════════════════════

async def jurisprudence(
    ctx: RunContext[MurdockDeps],
    query: str,
    court: str = "ambos",
) -> str:
    """Busca jurisprudência tributária nos tribunais superiores (STF e STJ).

    Pesquisa temas de repercussão geral (STF) e recursos repetitivos (STJ)
    em matéria tributária.

    Args:
        query: Tema jurídico (ex: "exclusão ICMS base PIS/COFINS", "DIFAL Simples Nacional")
        court: "stf", "stj" ou "ambos"
    """
    source_type = None
    if court == "stf":
        source_type = "fiscal_stf"
    elif court == "stj":
        source_type = "fiscal_stj"

    results = await hybrid_search(
        ctx.deps.db,
        f"jurisprudência tributária {query}",
        limit=6,
        source_type=source_type,
    )

    if not results:
        # Tentativa sem filtro de tipo
        results = await hybrid_search(
            ctx.deps.db, f"STF STJ tributário {query}", limit=6
        )

    if not results:
        return (
            f"Nenhuma jurisprudência encontrada para '{query}'. "
            "Recomendo consultar diretamente:\n"
            "  • STF: https://portal.stf.jus.br/jurisprudenciaRepercussao/\n"
            "  • STJ: https://processo.stj.jus.br/repetitivos/temas_repetitivos/"
        )

    parts = [f"═══ JURISPRUDÊNCIA — {court.upper()} ═══\n"]

    for i, r in enumerate(results, 1):
        src = r.get("source", {})
        parts.append(
            f"[{i}] {src.get('title', 'N/A')}\n"
            f"    Fundamentação: {src.get('fundamentacao', 'N/A')}\n"
            f"    Tribunal: {src.get('type', 'N/A')}\n"
            f"    Fonte: {src.get('url', 'N/A')}\n"
            f"    Trecho:\n    {r.get('content', '')[:500]}\n"
        )

    parts.append(
        "⚠ IMPORTANTE: Jurisprudência é dinâmica. Sempre verifique a vigência "
        "e eventuais modulações de efeitos nos portais oficiais."
    )

    return "\n".join(parts)
