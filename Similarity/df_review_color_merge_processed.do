* Load the CSV file
import delimited "df_review_color_merge_processed", clear

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

xtset product_id_num real_datetime

* Handle potential outliers
* For example, winsorize rating at 1% and 99%
gen rating_winsor = rating
egen rating_low = pctile(rating), p(1)
egen rating_high = pctile(rating), p(99)
replace rating_winsor = rating_low if rating < rating_low
replace rating_winsor = rating_high if rating > rating_high

// * Regression without interaction term
// xtreg rating_winsor volume valence variance cumulativetextlen cumulativetitlelen reviewer_expe i.mon, fe vce(cluster product_id_num)
//
// * Regression with treatment interaction (AfterTreat)
// xtreg rating_winsor aftertreat volume valence variance cumulativetextlen cumulativetitlelen reviewer_expe i.mon, fe vce(cluster product_id_num)

gen remainder = mod(month, 12)

* Adding interaction between similarity and AfterTreat to the model
gen similarity_x_aftertreat = color_similarity_corr_scaled * aftertreat
xtreg rating_winsor aftertreat similarity_x_aftertreat volume valence variance cumulativetextlen cumulativetitlelen reviewer_expe i.mon, fe vce(cluster product_id_num)

// * Exclude control groups and zero similarity scores
// gen treatment_active = aftertreat if color_similarity_bhattacharyya_s != 0
// replace treatment_active = 0 if aftertreat == 0
//
// * Create high and low similarity groups based on median of non-zero similarities
// su color_similarity_bhattacharyya_s if color_similarity_bhattacharyya_s != 0, detail
// gen high_group = (color_similarity_bhattacharyya_s >= r(p50)) if color_similarity_bhattacharyya_s != 0
//
// * Regression for the low similarity group
// xtreg rating_winsor aftertreat volume valence variance cumulativetextlen cumulativetitlelen reviewer_expe i.mon if high_group == 0, fe vce(cluster product_id_num)
//
// * Regression for the high similarity group
// xtreg rating_winsor aftertreat volume valence variance cumulativetextlen cumulativetitlelen reviewer_expe i.mon if high_group == 1, fe vce(cluster product_id_num)
