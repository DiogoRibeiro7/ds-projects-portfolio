# Run from projects/orange_storage_decision. Base R only.
source("R/sima_market_prices.R")

expect_error <- function(expression) {
  stopifnot(tryCatch({ force(expression); FALSE }, error = function(error) TRUE))
}

snapshot <- file.path(
  "data", "raw", "sima_oranges_wholesale_2025-01-01_2026-10-31.csv.gz"
)
stopifnot(file.exists(snapshot))

# The evidence is a real, unmodified GPP SIMA CSV export. Gzip is transport
# compression only: the original bytes and their SHA-256 are in MARKET_PRICES.md.
quotes <- read_sima_orange_prices(snapshot, market_stage = "wholesale")
stopifnot(
  nrow(quotes) == 3412L,
  identical(sort(unique(quotes$market)), c("MAC", "MAP", "MARL")),
  identical(sort(unique(quotes$species)), "Laranja"),
  identical(unique(quotes$market_stage), "wholesale"),
  identical(unique(quotes$price_unit), "EUR/kg"),
  identical(unique(quotes$source), "GPP SIMA"),
  min(quotes$quote_date) >= as.Date("2025-01-01"),
  max(quotes$quote_date) <= as.Date("2026-10-31"),
  all(is.finite(quotes$min_eur_per_kg)),
  all(quotes$min_eur_per_kg > 0),
  all(quotes$min_eur_per_kg <= quotes$frequent_eur_per_kg),
  all(quotes$frequent_eur_per_kg <= quotes$max_eur_per_kg),
  all(is.na(quotes$is_provisional)),
  !anyDuplicated(quotes[c("quote_date", "product_descriptor", "region", "market")])
)

# Product grading, origin and currency must not be lost when normalizing.
newhall <- quotes[
  quotes$quote_date == as.Date("2025-01-02") & quotes$market == "MAC" &
    grepl("Newhall\\*II\\*1, 2 e 3", quotes$product_descriptor),
  , drop = FALSE
]
stopifnot(nrow(newhall) == 1L,
          newhall$min_eur_per_kg == 0.60,
          newhall$max_eur_per_kg == 0.70,
          newhall$frequent_eur_per_kg == 0.65,
          grepl("Algarve\\*EUR/kg$", newhall$product_descriptor))

# A two-row fixture tests normalization and rejection paths independently
# of the archived dataset. Column names and decimal values match SIMA.
fixture <- data.frame(
  Produto = rep("Laranja*Newhall*II*Caixa*Algarve*EUR/kg", 2L),
  Espécie = rep("Laranja", 2L),
  Região = rep("Beira Litoral", 2L), Mercado = rep("MAC", 2L),
  Semana = c("1", "2"), Data = c("02-01-2025", "10-01-2025"),
  Mínima = c("0,60", "0.65"), Máxima = c("0,80", "0.85"),
  Freq = c("0,70", "0.75"), Provisórias = c("Não", "Sim"),
  check.names = FALSE, stringsAsFactors = FALSE
)

write_fixture <- function(value) {
  path <- tempfile(fileext = ".csv")
  utils::write.table(value, file = path, sep = ";", row.names = FALSE,
                     col.names = TRUE, quote = TRUE, na = "",
                     fileEncoding = "latin1")
  path
}

path <- write_fixture(fixture)
parsed <- read_sima_orange_prices(path, "wholesale")
stopifnot(nrow(parsed) == 2L, identical(parsed$is_provisional, c(FALSE, TRUE)),
          identical(parsed$frequent_eur_per_kg, c(0.70, 0.75)))
excluded <- read_sima_orange_prices(path, "wholesale", exclude_provisional = TRUE)
stopifnot(nrow(excluded) == 1L, !excluded$is_provisional)
unlink(path)

check_invalid <- function(changed) {
  path <- write_fixture(changed)
  on.exit(unlink(path))
  expect_error(read_sima_orange_prices(path, "wholesale"))
}

bad <- fixture
bad$Espécie[1] <- "Tangerina"
check_invalid(bad)
bad <- fixture
bad$Produto[1] <- "Laranja*Newhall*II*Caixa*Algarve*EUR/unidade"
check_invalid(bad)
bad <- fixture
bad$Mínima[1] <- "1.30"
check_invalid(bad)
bad <- fixture
bad$Mínima[1] <- "0,6.5"
check_invalid(bad)
bad <- fixture
bad$Data[1] <- "31-02-2025"
check_invalid(bad)
bad <- fixture
bad$Semana[1] <- "54"
check_invalid(bad)
bad <- fixture
bad$Provisórias[1] <- "maybe"
check_invalid(bad)
bad <- fixture
bad$Data[2] <- bad$Data[1]
check_invalid(bad)
bad <- fixture
bad$Freq <- NULL
check_invalid(bad)
bad <- fixture
bad$Provisórias <- NULL
path <- write_fixture(bad)
unknown <- read_sima_orange_prices(path, "wholesale",
                                   exclude_provisional = TRUE)
stopifnot(nrow(unknown) == 2L, all(is.na(unknown$is_provisional)))
unlink(path)

expect_error(read_sima_orange_prices("missing.csv", "wholesale"))
expect_error(read_sima_orange_prices(snapshot, "retail"))
expect_error(read_sima_orange_prices(snapshot, "production"))
expect_error(read_sima_orange_prices(snapshot, "wholesale", encoding = "bogus"))
expect_error(read_sima_orange_prices(snapshot, "wholesale",
                                    exclude_provisional = NA))

cat("Official SIMA orange-price ingestion tests passed.\n")
