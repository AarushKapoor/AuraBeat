AURABEAT
Project Design Document
Aarush Kapoor & Taylor Joasil


PROJECT OVERVIEW
AuraBeat is an interactive, gesture-controlled musical system that allows users to create and manipulate sound using hand movements captured through a standard webcam. By leveraging real-time hand tracking via MediaPipe Hands and applying mathematical and geometric analysis to that data, AuraBeat transforms gestures into musical outputs such as pitch, volume, and instrument selection.
This project explores alternative modes of musical interaction compared to traditional physical instruments. AuraBeat aims to provide a low-barrier, camera-based system that enables free-form musical experimentation while remaining responsive and intuitive.
Potential applications go beyond entertainment, such as education, accessibility, and creative performance. AuraBeat could be used to teach rhythm and musical concepts, serve as an accessibility-focused input method for users, or function as a sandbox for artists exploring new forms of digital musical expression.







SOFTWARE ARCHITECTURE
Hand Tracking and Input
MediaPipe Hands
Provides real-time detection of 21 hand landmarks per hand
Supplies normalized 2D/3D landmark coordinates
Serves as the primary input source for gesture recognition
OpenCV
Handles webcam input and frame capture
Integrates video frames into the MediaPipe processing pipeline
Mathematical and Gesture Processing
NumPy
Used for vectorized computation of distances, angles, velocities, and relative positions
Enables geometric analysis of hand posture and motion over time
Gesture features include:
Distances between fingers and palm
Joint angles and finger curvature
Hand velocity and acceleration
Relative hand position within the camera frame
Audio System
Pyo Audio Library
Provides real-time audio synthesis and playback
Supports continuous parameter control (pitch, amplitude)
Enables low-latency sound generation responsive to gesture input
Visual Interface
Kivy
Used to build the application UI
Displays visual feedback such as hand position indicators, gesture states, and audio mappings
Helps users understand how their gestures affect sound output
Development and Collaboration Tools
Programming Language: Python
IDE: Visual Studio Code
Version Control: Git, GitHub
Documentation: Google Docs
Team Communication: Discord

HARDWARE
At the current stage of development, no specialized hardware is required. All interaction is designed to function using commonly available consumer devices.
Desktop or laptop computers
Standard webcams (integrated or external)
Audio output devices (speakers or headphones)

DESIGN DECISIONS
AuraBeat follows a modular pipeline architecture:
Camera input and frame capture
Hand landmark detection via MediaPipe
Gesture and motion analysis layer
Mapping layer translating gestures to musical parameters
Audio synthesis and playback
Visual feedback and UI rendering
This separation allows individual components to be tested, refined, and replaced independently.






Gesture Mapping Strategy
Continuous gestures (e.g., hand height, distance between fingers) control continuous audio parameters
Discrete gestures (e.g., closed fist, extended fingers) trigger on/off events or mode changes
Smoothing and filtering are applied to reduce jitter and unintended triggers
Responsiveness
Low latency is treated as a primary design constraint. Gesture processing and audio generation are designed to operate in real time with minimal buffering to ensure immediate auditory feedback.

TESTING AND EVALUATION
AuraBeat will be evaluated for gesture accuracy, real-time performance, audio stability, and usability. Testing will be conducted throughout development to ensure the system remains responsive and intuitive.
Gesture Accuracy:
Hand landmark detection and gesture feature extraction will be tested under typical lighting and usage conditions. Our benchmark will be ≥ 95% successful hand detection when hands are in-frame, with gesture misclassification being ≤ 10%.
Latency and Performance:
End-to-end latency from gesture input to audio output will be measured to ensure real-time interaction. Our benchmark will be a gesture-to-audio latency ≤ 100 ms and stable performance at ≥ 24 FPS.
Audio Stability:
Continuous gesture mappings will be evaluated for smooth parameter changes without unintentional audio effects. Our benchmark will be no audible glitches during continuous control and ≥ 90% reliable triggering of specific gestures and combinations.
Visual Feedback and Usability:
Visual indicators will be validated for clarity and synchronization with gesture and audio output. Informal user testing will assess intuitiveness. Our benchmark for visual feedback is synchronization within one frame of audio output
Robustness:
System behavior will be tested during hand loss, occlusion, and re-entry. Our benchmark will be hand tracking recovery within 1 second and no system crashes during normal use.
DESIGN DECISIONS YET TO BE MADE
Final set of supported gestures and their exact mappings
Whether to support one-hand or two-hand interaction modes
Selection and number of available instruments or sound types
Inclusion and scope of a rhythm-based or game-like interaction mode

PROJECT PHASES
Research and environment setup
Real-time hand tracking implementation
Gesture feature extraction and analysis
Gesture-to-audio mapping
Audio engine integration
Visual feedback and UI design
System testing and refinement
Documentation and final presentation

MILESTONES
MediaPipe hand tracking prototype
Basic gesture detection and analysis
Audio synthesis integration
Gesture-controlled sound demo
Visual interface implementation
Full system integration
Final demonstration and documentation submission

PROJECT GOALS & DELIVERABLES
Main Goals
Accurately track hand movements in real time
Translate gestures into expressive musical output
Maintain low latency and system stability
Provide clear visual feedback for usability
Deliverables
Tangible Deliverables
Fully functional gesture-controlled music application
Source code repository
User guide describing interaction methods
Technical documentation explaining system architecture and gesture mappings
Other Deliverables
Exploration of gesture-based musical interaction design
Insights into real-time motion analysis and audio responsiveness
Additional Goals (Outside Classroom Scope)
Expanded instrument library and sound design
Preset saving and replay functionality
Multi-user or collaborative performance modes
Public exhibition or performance-oriented features

UNKNOWNS & RISKS
Potential Risks
Gesture ambiguity leading to unintended sound output
Latency issues affecting user experience
Sensitivity to lighting conditions or camera quality
Difficulty balancing expressiveness with simplicity
Mitigation Strategies
Iterative testing and gesture refinement
Smoothing and thresholding of gesture input
User feedback-driven adjustments
Clear visual feedback to reinforce gesture-sound relationships



DIAGRAMS
Architecture Diagram





Sequence Diagram



Use Case Diagram





This design document defines the architecture, scope, and goals of AuraBeat while allowing flexibility for iteration and refinement throughout development.
