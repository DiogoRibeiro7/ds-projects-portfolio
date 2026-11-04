# Run from projects/orange_storage_decision. Base R only.
source("R/release_audit.R")

expect_error <- function(expression) {
  stopifnot(tryCatch({ force(expression); FALSE }, error = function(error) TRUE))
}

# The archived reports, machine-readable evidence, local links and selected
# figures quoted in the scientific synthesis must remain internally consistent.
audit <- audit_orange_release(".")
stopifnot(is.list(audit), audit$archived_studies == 14L,
          audit$csv_files >= 14L, audit$markdown_documents >= 24L,
          audit$verified_local_links > 0L, audit$verified_headlines == 17L)

# Bad input and missing local links should fail clearly rather than be skipped.
expect_error(audit_orange_release(NA_character_))
expect_error(audit_orange_release(1))
expect_error(audit_orange_release(""))
expect_error(audit_orange_release("a-release-directory-that-does-not-exist"))
expect_error(check_orange_markdown_links("a-missing-file.md"))

# Check the link parser independently, including an image and a URL/anchor that
# must not be mistaken for relative files. All test files are temporary.
scratch <- tempfile("orange-release-link-")
dir.create(scratch)
document <- file.path(scratch, "example.md")
writeLines(c("[local](value.csv) ![plot](figure.png)",
             "[external](https://example.com) [anchor](#section)"), document)
stopifnot(file.create(file.path(scratch, "value.csv")))
stopifnot(file.create(file.path(scratch, "figure.png")))
stopifnot(check_orange_markdown_links(document) == 2L)
writeLines("[broken](missing.csv)", document)
expect_error(check_orange_markdown_links(document))
unlink(file.path(scratch, c("example.md", "value.csv", "figure.png")))
unlink(scratch, recursive = FALSE)

# A copied headline, but from the wrong result row or with a changed amount,
# must not pass the release evidence check.
expect_error(check_orange_headline_metric(
  ".", "price_sensitivity", "summary.csv",
  list(scenario = "perfect_forecast"), "stratified_minus_immediate_eur", 999
))
expect_error(check_orange_headline_metric(
  ".", "price_sensitivity", "summary.csv",
  list(scenario = "does_not_exist"), "stratified_minus_immediate_eur", 13.95
))

cat(sprintf("Release audit passed: %d archived studies, %d CSV files, %d ",
            audit$archived_studies, audit$csv_files, audit$markdown_documents),
    "Markdown documents, ", audit$verified_local_links, " local links and ",
    audit$verified_headlines, " headline figures.\n", sep = "")
