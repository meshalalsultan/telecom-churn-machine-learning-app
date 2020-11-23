import streamlit as st 


"""
This is streamlit app
"""



genre = st.radio(
	"Chose if you like or what ?",
	('like','dislike'))

	if genre == 'like':
			st.write('Thank Yo :)')
	else:
			st.write("Ok , i dislike you eaither")