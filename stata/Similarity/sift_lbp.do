* Load the CSV file
import delimited "df_review_sift_lbp", clear

gen real_datetime = clock(datetime, "DMYhms")
format real_datetime %tc

encode product_id, gen(product_id_num)

* Check for duplicates again with more detailed listing
bysort product_id_num real_datetime: gen dup_id = _n
list product_id_num real_datetime dup_id if dup_id > 1, noobs

* Drop duplicates where dup_id is 2
drop if dup_id == 2

* Drop the dup_id variable as it is no longer needed
drop dup_id

* Handle potential outliers
* For example, winsorize rating at 1% and 99%
gen rating_winsor = rating
egen rating_low = pctile(rating), p(1)
egen rating_high = pctile(rating), p(99)
replace rating_winsor = rating_low if rating < rating_low
replace rating_winsor = rating_high if rating > rating_high

xtset product_id_num real_datetime

gen remainder = mod(month, 12)

// * Regression without interaction term
// xtreg rating volume valence variance cumulativetextlen cumulativetitlelen reviewer_expe i.mon, fe vce(cluster product_id_num)
//
// * Regression with treatment interaction (AfterTreat)
// xtreg rating aftertreat volume valence variance cumulativetextlen cumulativetitlelen reviewer_expe i.mon, fe vce(cluster product_id_num)

* Adding interaction between similarity and AfterTreat to the model
gen similarity_x_aftertreat = lbp_similarity * aftertreat
xtreg rating_winsor aftertreat similarity_x_aftertreat volume valence variance cumulativetextlen cumulativetitlelen reviewer_expe i.mon, fe vce(cluster product_id_num)
