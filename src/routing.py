"""Provisional, project-created routes for the 77 banking categories."""

from __future__ import annotations

from dataclasses import dataclass


TEAM_CATEGORIES = {
    "card_support": {
        "activate_my_card", "apple_pay_or_google_pay", "card_about_to_expire",
        "card_acceptance", "card_arrival", "card_delivery_estimate", "card_linking",
        "card_not_working", "card_swallowed", "change_pin", "compromised_card",
        "contactless_not_working", "disposable_card_limits", "get_disposable_virtual_card",
        "get_physical_card", "getting_spare_card", "getting_virtual_card",
        "lost_or_stolen_card", "order_physical_card", "pin_blocked",
        "virtual_card_not_working", "visa_or_mastercard",
    },
    "payments": {
        "card_payment_fee_charged", "card_payment_not_recognised",
        "card_payment_wrong_exchange_rate", "declined_card_payment",
        "direct_debit_payment_not_recognised", "extra_charge_on_statement",
        "pending_card_payment", "refund_not_showing_up", "request_refund",
        "reverted_card_payment", "transaction_charged_twice",
    },
    "transfers": {
        "balance_not_updated_after_bank_transfer", "beneficiary_not_allowed",
        "cancel_transfer", "declined_transfer", "failed_transfer", "pending_transfer",
        "receiving_money", "transfer_fee_charged", "transfer_into_account",
        "transfer_not_received_by_recipient", "transfer_timing",
    },
    "cash_atm": {
        "atm_support", "balance_not_updated_after_cheque_or_cash_deposit",
        "cash_withdrawal_charge", "cash_withdrawal_not_recognised",
        "declined_cash_withdrawal", "pending_cash_withdrawal",
        "wrong_amount_of_cash_received", "wrong_exchange_rate_for_cash_withdrawal",
    },
    "account_identity": {
        "age_limit", "country_support", "edit_personal_details",
        "fiat_currency_support", "lost_or_stolen_phone", "passcode_forgotten",
        "terminate_account", "unable_to_verify_identity", "verify_my_identity",
        "verify_source_of_funds", "why_verify_identity",
    },
    "top_up": {
        "automatic_top_up", "pending_top_up", "top_up_by_bank_transfer_charge",
        "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed",
        "top_up_limits", "top_up_reverted", "topping_up_by_card", "verify_top_up",
    },
    "foreign_exchange": {
        "exchange_charge", "exchange_rate", "exchange_via_app",
        "supported_cards_and_currencies",
    },
}

URGENT_CATEGORIES = {
    "card_payment_not_recognised", "cash_withdrawal_not_recognised",
    "compromised_card", "direct_debit_payment_not_recognised",
    "extra_charge_on_statement", "lost_or_stolen_card", "lost_or_stolen_phone",
    "transaction_charged_twice",
}

STANDARD_CATEGORIES = {
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_not_working", "card_swallowed",
    "declined_card_payment", "declined_cash_withdrawal", "declined_transfer",
    "failed_transfer", "passcode_forgotten", "pending_card_payment",
    "pending_cash_withdrawal", "pending_top_up", "pending_transfer", "pin_blocked",
    "refund_not_showing_up", "request_refund", "reverted_card_payment",
    "top_up_failed", "top_up_reverted", "transfer_not_received_by_recipient",
    "unable_to_verify_identity", "verify_source_of_funds",
    "virtual_card_not_working", "wrong_amount_of_cash_received",
}


@dataclass(frozen=True)
class Route:
    priority: str
    team: str


def build_routes(allowed_categories: set[str]) -> dict[str, Route]:
    """Require exact coverage so a changed label set cannot inherit a silent default."""
    grouped = [category for categories in TEAM_CATEGORIES.values() for category in categories]
    if len(grouped) != len(set(grouped)) or set(grouped) != allowed_categories:
        raise ValueError("routing teams must cover each approved category exactly once")
    if URGENT_CATEGORIES & STANDARD_CATEGORIES:
        raise ValueError("routing priority groups overlap")
    if not (URGENT_CATEGORIES | STANDARD_CATEGORIES) <= allowed_categories:
        raise ValueError("routing priority contains an unknown category")

    return {
        category: Route(
            priority=(
                "urgent" if category in URGENT_CATEGORIES
                else "standard" if category in STANDARD_CATEGORIES
                else "low"
            ),
            team=team,
        )
        for team, categories in TEAM_CATEGORIES.items()
        for category in categories
    }
