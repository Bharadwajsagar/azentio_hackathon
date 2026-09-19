"""Thin wrapper around utils.io -- kept as its own service so controllers don't
need to know about filesystem paths directly."""

from utils.io import load_raw_csvs
from utils.logging import get_logger

logger = get_logger(__name__)


def load_all():
    logger.info("Loading raw CSVs (transactions, accounts, customers)...")
    tx, acc, cust = load_raw_csvs()
    logger.info(
        "Loaded: transactions=%d rows, accounts=%d rows, customers=%d rows",
        len(tx),
        len(acc),
        len(cust),
    )
    return tx, acc, cust
