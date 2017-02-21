packages <- c("jsonlite", "dplyr", "purrr")
purrr::walk(packages, library, character.only = TRUE, warn.conflicts = FALSE)
train <- fromJSON("../../data/train.json")
vars <- setdiff(names(train), c("photos", "features"))
train <- map_at(train, vars, unlist) %>% tibble::as_tibble(.) %>%
  select(longitude, latitude, listing_id, building_id, interest_level)
test <- fromJSON("../../data/test.json")
vars <- setdiff(names(test), c("photos", "features"))
test <- map_at(test, vars, unlist) %>% tibble::as_tibble(.) %>%
  select(longitude, latitude, listing_id, building_id)

library(shiny)
library(leaflet)

# Define UI for application that draws a histogram
ui <- fluidPage(
  tags$head(
    tags$link(rel = "stylesheet", type = "text/css", href = "custom.css")
  ),
  leafletOutput('map',width="100%",height="1000px")
)

# Define server logic required to draw a histogram
server <- function(input, output, session) {
   output$map <- renderLeaflet({
     map <- leaflet() %>% addTiles()
     overlayGroups <- c('train_all')
     map <- map %>% addMarkers(data = train, group = paste('train', 'all', sep = '_'),
                               clusterOptions = markerClusterOptions(),
                               popup = ~paste(listing_id, building_id, interest_level))
     for(level in c('low', 'medium', 'high')) {
       map <- map %>% addMarkers(data = train %>% filter(interest_level == level),
                                 group = paste('train', level, sep = '_'),
                                 clusterOptions = markerClusterOptions(),
                                 popup = ~paste(listing_id, building_id, interest_level))
       overlayGroups <- c(overlayGroups, paste('train', level, sep = '_'))
     }
     map <- map %>% addMarkers(data = test, group = 'test',
                               clusterOptions = markerClusterOptions(),
                               popup = ~paste(listing_id, building_id))
     overlayGroups <- c(overlayGroups, 'test')
     map %>% addLayersControl(baseGroups = overlayGroups,
                              options = layersControlOptions(collapsed = F))
   })
}

# Run the application 
shinyApp(ui = ui, server = server)

