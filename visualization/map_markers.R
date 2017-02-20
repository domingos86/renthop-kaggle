packages <- c("jsonlite", "dplyr", "purrr")
purrr::walk(packages, library, character.only = TRUE, warn.conflicts = FALSE)
data <- fromJSON("../data/train.json")
vars <- setdiff(names(data), c("photos", "features"))
data <- map_at(data, vars, unlist) %>% tibble::as_tibble(.)

library(leaflet)
leaflet(data = data) %>% addTiles() %>%
  addMarkers(clusterOptions = markerClusterOptions(),
             popup = ~paste(listing_id, building_id, interest_level))