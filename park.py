import streamlit as st

if 'parking_data' not in st.session_state:
    st.session_state.parking_data = {
        'Erode': {'Total Spaces': 50, 'Available Spaces': 20, 'booked_cars': []},
        'Tiruppur': {'Total Spaces': 30, 'Available Spaces': 10, 'booked_cars': []},
        'Coimbatore': {'Total Spaces': 40, 'Available Spaces': 25, 'booked_cars': []},
    }

def show_homepage():
    st.title("Nearby Parking Lots")

    cols = st.columns(3)  
    for i, (location, data) in enumerate(st.session_state.parking_data.items()):
        with cols[i]:
            st.header(location)
            st.write(f"*Total Spaces*: {data['Total Spaces']}")
            st.write(f"*Available Spaces*: {data['Available Spaces']}")
            st.progress(data['Available Spaces'] / data['Total Spaces']) 
            if st.button(f"Select {location}", key=f"select_{location}"):
                st.session_state.selected_location = location
                st.session_state.page = 'booking'
                st.rerun()  

def show_booking_page():
    selected_location = st.session_state.selected_location
    st.title(f"Booking - {selected_location}")

    location_data = st.session_state.parking_data[selected_location]
    total_spaces = location_data['Total Spaces']
    available_spaces = location_data['Available Spaces']
    
    st.write(f"*Total Spaces:* {total_spaces}")
    st.write(f"*Available Spaces:* {available_spaces}")
    st.progress(available_spaces / total_spaces)  

    car_number = st.text_input("Enter your car number to book:")
    
    if car_number:
        if car_number in location_data['booked_cars']:
            st.error("This car number has already booked a space!")
        else:
            if available_spaces > 0 and st.button("Book a Space"):
                st.session_state.parking_data[selected_location]['Available Spaces'] -= 1
                st.session_state.parking_data[selected_location]['booked_cars'].append(car_number)
                st.success("Space booked successfully!")
                st.balloons() 
                st.rerun()

    st.write("## Leave a Parking Space")
    if location_data['booked_cars']:
        booked_car = st.selectbox("Select your booked car number to leave:", location_data['booked_cars'], key="leave_car")
        if st.button("Leave Space"):
            st.session_state.parking_data[selected_location]['Available Spaces'] += 1
            st.session_state.parking_data[selected_location]['booked_cars'].remove(booked_car)
            st.success(f"Space vacated for car number {booked_car}!")
            st.rerun()
    else:
        st.info("No cars currently booked.")

    if st.button("View Location on Google Maps"):
        location_url = f"https://www.google.com/maps/search/{selected_location}"
        st.markdown(f"[Click here to open Google Maps for {selected_location}]({location_url})", unsafe_allow_html=True)

    if st.button("Back to Home"):
        st.session_state.page = 'home'
        st.rerun()

if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_location' not in st.session_state:
    st.session_state.selected_location = None

if st.session_state.page == 'home':
    show_homepage()
elif st.session_state.page == 'booking':
    show_booking_page()