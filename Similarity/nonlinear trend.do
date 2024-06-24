* Load the CSV file
import delimited "df_review_processed.csv", clear
// import delimited "reviews_clean_processed.csv", clear

* Generate a datetime variable
gen real_datetime = clock(datetime, "DMYhms")
format real_datetime %tc

* Encode product_id as a numeric variable
encode product_id, gen(product_id_num)

* Check for duplicates again with more detailed listing
bysort product_id_num real_datetime: gen dup_id = _n
list product_id_num real_datetime dup_id if dup_id > 1, noobs

* Drop duplicates where dup_id is 2
drop if dup_id == 2

* Drop the dup_id variable as it is no longer needed
drop dup_id

* Set panel data structure
xtset product_id_num real_datetime

// * Generate linear time trend for each product
// by product_id_num (real_datetime), sort: gen linear_time_trend = _n
//
// * Generate nonlinear (quadratic) time trend for each product
// gen quadratic_time_trend = linear_time_trend^2
//
// * Run the regression with linear and quadratic time trends
// xtreg rating aftertreat volume valence variance cumulativetextlen cumulativetitlelen reviewer_expe i.mon linear_time_trend, fe vce(cluster product_id_num)
//
// xtreg rating aftertreat volume valence variance cumulativetextlen cumulativetitlelen reviewer_expe i.mon linear_time_trend quadratic_time_trend, fe vce(cluster product_id_num)

// *ordered logit model
// xtologit rating aftertreat volume valence variance cumulativetextlen cumulativetitlelen reviewer_expe i.mon, vce(cluster product_id_num)
//
// * Marginal effects for outcome category 1
// margins, dydx(aftertreat) predict(outcome(1))
//
// * Marginal effects for outcome category 2
// margins, dydx(aftertreat) predict(outcome(2))
//
// * Marginal effects for outcome category 3
// margins, dydx(aftertreat) predict(outcome(3))
//
// * Marginal effects for outcome category 4
// margins, dydx(aftertreat) predict(outcome(4))
//
// * Marginal effects for outcome category 5
// margins, dydx(aftertreat) predict(outcome(5))



* Assume CGI_date and review_date are already in Stata date format
gen first_cgi_month = month(first_cgi_date) if first_cgi == 1  // Create a variable for the first CGI month
bysort product_id_num (CGI_date): gen first_cgi_date = CGI_date[1]  // Get the first CGI date for each product

* Create early and late treatment indicators
gen early_treated = (first_cgi_month == 1)
gen late_treated = (first_cgi_month > 1)
