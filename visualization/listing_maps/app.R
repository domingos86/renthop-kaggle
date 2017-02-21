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
     leaflet() %>% addTiles() %>%
       addMarkers(data = train, group = 'train', clusterOptions = markerClusterOptions(),
                  popup = ~paste(listing_id, building_id, interest_level)) %>%
       addMarkers(data = test, group = 'test', clusterOptions = markerClusterOptions(),
                  popup = ~paste(listing_id, building_id)) %>%
       addLayersControl(baseGroups = c('train', 'test'),
                        options = layersControlOptions(collapsed = F))
   })
}

# Run the application 
shinyApp(ui = ui, server = server)

