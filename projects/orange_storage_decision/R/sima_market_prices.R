# Official Portuguese SIMA orange-market quotations, independent of the
# completed synthetic orange-storage model. Uses base R only.

#' Require a nonempty character scalar.
#' @param value Value supplied by the caller.
#' @param label Human-readable argument name.
#' @return The supplied scalar, invisibly.
assert_sima_string <- function(value, label) {
  if (!is.character(value) || length(value) != 1L || is.na(value) ||
      !nzchar(trimws(value))) {
    stop(sprintf("%s must be one nonempty string", label), call. = FALSE)
  }
  invisible(value)
}

#' Map a SIMA column heading to an ASCII comparison key.
#' @param headings Original CSV header labels, including Portuguese accents.
#' @return Lowercase ASCII keys with whitespace and punctuation removed.
sima_heading_keys <- function(headings) {
  headings <- sub("^\ufeff", "", headings)
  ascii <- iconv(headings, from = "", to = "ASCII//TRANSLIT")
  if (anyNA(ascii)) {
    stop("SIMA headings cannot be decoded with the selected encoding",
         call. = FALSE)
  }
  tolower(gsub("[^a-zA-Z0-9]", "", ascii))
}

#' Resolve one required or optional SIMA column.
#' @param keys Normalized heading keys.
#' @param aliases Accepted headings for this field.
#' @param label Standardized field name used in diagnostic messages.
#' @param required Whether exactly one matching heading is required.
#' @return One column index, or NA_integer_ when the field is optional.
sima_column_index <- function(keys, aliases, label, required = TRUE) {
  matches <- which(keys %in% aliases)
  if (length(matches) > 1L || (required && length(matches) != 1L)) {
    stop(sprintf("SIMA export needs exactly one %s column", label),
         call. = FALSE)
  }
  if (length(matches) == 0L) NA_integer_ else as.integer(matches)
}

#' Parse a positive EUR/kg price without locale-dependent decimal inference.
#' @param values Raw text values from a quotation column.
#' @param label Name of the source quotation column.
#' @return Positive numeric quotations, preserving their original scale.
parse_sima_price <- function(values, label) {
  raw <- trimws(as.character(values))
  valid <- !is.na(raw) & grepl("^[0-9]+([.,][0-9]+)?$", raw)
  if (!all(valid)) {
    stop(sprintf("%s contains missing, ambiguous or invalid prices", label),
         call. = FALSE)
  }
  prices <- suppressWarnings(as.numeric(sub(",", ".", raw, fixed = TRUE)))
  if (anyNA(prices) || any(!is.finite(prices)) || any(prices <= 0)) {
    stop(sprintf("%s must contain finite, strictly positive prices", label),
         call. = FALSE)
  }
  prices
}

#' Parse calendar dates without silently accepting invalid dates.
#' @param values Raw quotation dates in day-first or ISO notation.
#' @return A Date vector, rejecting mixed malformed rows.
parse_sima_date <- function(values) {
  raw <- trimws(as.character(values))
  parsed <- as.Date(rep(NA_character_, length(raw)))
  for (date_format in c("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d")) {
    candidate <- suppressWarnings(as.Date(raw, format = date_format))
    matches <- !is.na(candidate) & format(candidate, date_format) == raw
    parsed[matches] <- candidate[matches]
  }
  if (anyNA(parsed)) {
    stop("SIMA export contains missing or invalid quotation dates",
         call. = FALSE)
  }
  parsed
}

#' Interpret SIMA provisional markers without inventing missing information.
#' @param markers Optional source field. NULL means no such field was exported.
#' @param n_rows Number of source quotations.
#' @return Logical values: TRUE provisional, FALSE final, NA unknown.
parse_sima_provisional <- function(markers, n_rows) {
  if (is.null(markers)) return(rep(NA, n_rows))
  raw <- trimws(as.character(markers))
  ascii <- iconv(raw, from = "", to = "ASCII//TRANSLIT")
  if (anyNA(ascii[!is.na(raw)])) {
    stop("Cannot decode SIMA provisional indicators", call. = FALSE)
  }
  flags <- tolower(ascii)
  yes <- flags %in% c("sim", "s", "yes", "y", "true", "1")
  no <- flags %in% c("nao", "n", "no", "false", "0")
  unknown <- is.na(flags) | flags == ""
  if (!all(yes | no | unknown)) {
    stop("SIMA export contains unrecognized provisional indicators",
         call. = FALSE)
  }
  result <- rep(NA, n_rows)
  result[yes] <- TRUE
  result[no] <- FALSE
  result
}

#' Ingest official GPP SIMA orange quotations from a CSV or CSV.gz export.
#'
#' The checked-in 2025-2026 export comes from the *Mercados Abastecedores*
#' (wholesale) interface, not farm-gate or retail transactions. SIMA reports
#' minimum, maximum and most-frequent quoted prices; it does not report
#' executed sales or storage-lot revenues. The original Produto descriptor is
#' retained to prevent pooling varieties, grades, packaging and origins.
#'
#' @param path Path to an original SIMA CSV export, optionally gzip-compressed.
#' @param market_stage Must be "wholesale" for the verified GPP export;
#'   production-market imports require separate source validation.
#' @param encoding File encoding: the verified 2026 export uses "latin1".
#' @param exclude_provisional Remove only rows explicitly labelled provisional;
#'   an absent indicator remains unknown, never assumed final.
#' @return Data frame with source date, week, descriptor, location, EUR/kg
#'   quotations, provisional status and file/source provenance. Does not
#'   interpolate, aggregate, fit forecasts or modify the synthetic study.
read_sima_orange_prices <- function(path, market_stage, encoding = "latin1",
                                    exclude_provisional = FALSE) {
  assert_sima_string(path, "path")
  assert_sima_string(market_stage, "market_stage")
  if (market_stage != "wholesale") {
    stop("Only verified SIMA wholesale exports are supported", call. = FALSE)
  }
  assert_sima_string(encoding, "encoding")
  if (!encoding %in% c("latin1", "UTF-8")) {
    stop("encoding must be 'latin1' or 'UTF-8'", call. = FALSE)
  }
  if (!is.logical(exclude_provisional) || length(exclude_provisional) != 1L ||
      is.na(exclude_provisional)) {
    stop("exclude_provisional must be TRUE or FALSE", call. = FALSE)
  }
  if (!file.exists(path) || dir.exists(path)) {
    stop("SIMA export file does not exist", call. = FALSE)
  }

  connection <- if (grepl("\\.gz$", path, ignore.case = TRUE)) {
    gzfile(path, open = "rt", encoding = encoding)
  } else {
    file(path, open = "rt", encoding = encoding)
  }
  on.exit(close(connection), add = TRUE)
  raw <- tryCatch(utils::read.table(
    connection, header = TRUE, sep = ";", quote = "\"", dec = ".",
    colClasses = "character", check.names = FALSE, comment.char = "",
    strip.white = TRUE, stringsAsFactors = FALSE
  ), error = function(error) {
    stop(sprintf("Cannot parse semicolon-delimited SIMA export: %s",
                 conditionMessage(error)), call. = FALSE)
  })
  if (nrow(raw) < 1L) {
    stop("SIMA export has no quotation rows", call. = FALSE)
  }

  keys <- sima_heading_keys(names(raw))
  columns <- list(
    product = sima_column_index(keys, "produto", "product"),
    species = sima_column_index(keys, "especie", "species"),
    region = sima_column_index(keys, "regiao", "region"),
    market = sima_column_index(keys, "mercado", "market"),
    week = sima_column_index(keys, "semana", "week"),
    date = sima_column_index(keys, c("data", "iniciosemana", "dia"), "date"),
    minimum = sima_column_index(keys, c("minima", "min"), "minimum price"),
    maximum = sima_column_index(keys, c("maxima", "max"), "maximum price"),
    frequent = sima_column_index(keys, c("freq", "frequente"), "frequent price"),
    provisional = sima_column_index(
      keys, c("provisorias", "provisoria", "provisional"),
      "provisional", required = FALSE
    )
  )
  field <- function(name) trimws(as.character(raw[[columns[[name]]]]))
  product <- field("product")
  species <- field("species")
  region <- field("region")
  market <- field("market")
  week_raw <- field("week")
  quote_date <- parse_sima_date(field("date"))

  if (anyNA(product) || anyNA(species) || anyNA(region) || anyNA(market) ||
      any(!nzchar(product) | !nzchar(species) |
          !nzchar(region) | !nzchar(market))) {
    stop("SIMA product, species, region and market cannot be blank",
         call. = FALSE)
  }
  if (any(tolower(species) != "laranja") ||
      any(!grepl("^Laranja\\*", product, ignore.case = TRUE))) {
    stop("SIMA export must contain orange quotations only", call. = FALSE)
  }
  units <- sub("^.*\\*", "", product)
  if (any(units != "EUR/kg")) {
    stop("Unexpected SIMA price unit: expected EUR/kg in each product descriptor",
         call. = FALSE)
  }
  if (anyNA(week_raw) || any(!grepl("^[0-9]{1,2}$", week_raw))) {
    stop("SIMA week labels must be whole numbers", call. = FALSE)
  }
  week <- as.integer(week_raw)
  if (any(week < 1L | week > 53L)) {
    stop("SIMA week labels must lie between 1 and 53", call. = FALSE)
  }

  minimum <- parse_sima_price(field("minimum"), "minimum")
  maximum <- parse_sima_price(field("maximum"), "maximum")
  frequent <- parse_sima_price(field("frequent"), "frequent")
  if (any(minimum > frequent | frequent > maximum)) {
    stop("Each frequent price must be between the minimum and maximum quote",
         call. = FALSE)
  }
  provisional <- parse_sima_provisional(
    if (is.na(columns$provisional)) NULL else field("provisional"),
    nrow(raw)
  )

  # Several graded products may share a date/market: only exact quote keys
  # are duplicates. Do not aggregate different origin, cultivar or packaging.
  quote_key <- data.frame(quote_date, product, region, market)
  if (anyDuplicated(quote_key)) {
    stop("Duplicate SIMA quotation key: date, product, region and market",
         call. = FALSE)
  }
  retain <- if (exclude_provisional) is.na(provisional) | !provisional else
    rep(TRUE, nrow(raw))
  if (!any(retain)) {
    stop("No orange quotations remain after the provisional filter",
         call. = FALSE)
  }

  normalized <- data.frame(
    quote_date = quote_date, sima_week = week,
    product_descriptor = product, species = species, region = region,
    market = market, market_stage = market_stage, price_unit = units,
    min_eur_per_kg = minimum, max_eur_per_kg = maximum,
    frequent_eur_per_kg = frequent, is_provisional = provisional,
    source = "GPP SIMA",
    source_url = "https://regsima.gpp.pt/regsima/consulta/mercados?tm=11",
    source_file = basename(path),
    source_file_md5 = unname(tools::md5sum(path)),
    stringsAsFactors = FALSE
  )
  normalized <- normalized[retain, , drop = FALSE]
  normalized <- normalized[
    order(normalized$quote_date, normalized$market,
          normalized$product_descriptor, method = "radix"), , drop = FALSE
  ]
  rownames(normalized) <- NULL
  normalized
}
