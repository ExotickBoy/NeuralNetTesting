package core;

import static java.lang.Math.hypot;
import static java.lang.Math.pow;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.HashMap;
import java.util.function.Consumer;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

public class NetworkTest {
	
	public static void main(String[] args) {
		
		NeuralNetwork network = NeuralNetwork.load(new File("nets/network.nwk"));
		
		JFrame frame = new JFrame("Network Test");
		frame.setContentPane(new MainPane(network));
		frame.pack();
		frame.setLocationRelativeTo(null);
		// frame.setResizable(false);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);
		
	}
	
	private static class MainPane extends JPanel {
		
		private static final long serialVersionUID = 1L;
		
		private NeuralNetwork network;
		
		private JPanel bottomPanel;
		private JButton resetButton;
		private ImageEditor imageEditor;
		private JLabel result;
		
		public MainPane(NeuralNetwork network) {
			
			this.network = network;
			
			imageEditor = new ImageEditor();
			imageEditor.setCallback(this::callback);
			
			bottomPanel = new JPanel();
			bottomPanel.setLayout(new BorderLayout());
			result = new JLabel("I think its a ");
			bottomPanel.add(result);
			
			resetButton = new JButton("Reset");
			resetButton.setToolTipText("if you press this button the current image will get completly cucked");
			resetButton.addActionListener((e) -> {
				
				imageEditor.reset();
				
			});
			bottomPanel.add(resetButton, BorderLayout.NORTH);
			
			setLayout(new BorderLayout());
			add(imageEditor, BorderLayout.NORTH);
			add(bottomPanel, BorderLayout.SOUTH);
			
		}
		
		private void callback(BufferedImage i) {
			
			Matrix img = new Matrix(1, 28 * 28);
			
			int k = 0;
			
			for (int x = 0; x < 28; x++) {
				for (int y = 0; y < 28; y++) {
					
					img.set(k++, (i.getRGB(x, y) & 0xff) / 255d);
					
				}
			}
			
			Matrix result = network.forward(img);
			System.out.println(result);
			HashMap<Integer, Double> map = new HashMap<>();
			for (int j = 0; j < 10; j++) {
				map.put(j, result.get(0, j));
			}
			map.entrySet().stream().max((a, b) -> Double.compare(a.getValue(), b.getValue())).ifPresent(e -> {
				
				this.result.setText("I'm " + String.format("%.2f", e.getValue() * 100) + "% sure that it's a " + e.getKey());
				
			});
			
		}
		
	}
	
	private static class ImageEditor extends JPanel implements MouseMotionListener, MouseListener {
		
		private static final long serialVersionUID = 1L;
		
		private static final int IMAGE_WIDTH = 28;
		private static final int IMAGE_HEIGHT = 28;
		private static final int SCALE = 10;
		
		private BufferedImage img;
		private Consumer<BufferedImage> callback;
		
		public ImageEditor() {
			
			img = new BufferedImage(IMAGE_WIDTH, IMAGE_HEIGHT, BufferedImage.TYPE_INT_ARGB);
			makeMark(-100, -100);
			
			setPreferredSize(new Dimension(IMAGE_WIDTH * SCALE, IMAGE_HEIGHT * SCALE));
			addMouseListener(this);
			addMouseMotionListener(this);
			
		}
		
		public void setCallback(Consumer<BufferedImage> callback) {
			
			this.callback = callback;
			
		}
		
		public void reset() {
			
			img = new BufferedImage(IMAGE_WIDTH, IMAGE_HEIGHT, BufferedImage.TYPE_INT_ARGB);
			makeMark(-100, -100);
			
			if (callback != null) {
				callback.accept(img);
			}
			
		}
		
		@Override
		protected void paintComponent(Graphics g) {
			
			super.paintComponent(g);
			
			g.drawImage(img, 0, 0, IMAGE_WIDTH * SCALE, IMAGE_HEIGHT * SCALE, null);
			
		}
		
		private void makeMark(double mx, double my) {
			
			for (int x = 0; x < IMAGE_WIDTH; x++) {
				for (int y = 0; y < IMAGE_HEIGHT; y++) {
					
					double dis = hypot(x - mx, y - my);
					
					double i = distanceFunction(dis, 1.5);
					img.setRGB(x, y, img.getRGB(x, y) | (((int) (i * 255)) | ((int) (i * 255)) << 8 | ((int) (i * 255)) << 16 | 255 << 24));
					
				}
			}
			repaint();
			
		}
		
		private double distanceFunction(double d, double r) {
			
			return 1 / (1 + pow(1000, d - r));
			
		}
		
		@Override
		public void mouseDragged(MouseEvent e) {
			
			makeMark(e.getX() / (double) SCALE, e.getY() / (double) SCALE);
			
		}
		
		@Override
		public void mouseClicked(MouseEvent e) {
			
			makeMark(e.getX() / (double) SCALE, e.getY() / (double) SCALE);
			
		}
		
		@Override
		public void mouseMoved(MouseEvent e) {}
		
		@Override
		public void mouseEntered(MouseEvent e) {}
		
		@Override
		public void mouseExited(MouseEvent e) {}
		
		@Override
		public void mousePressed(MouseEvent e) {}
		
		@Override
		public void mouseReleased(MouseEvent e) {
			
			if (callback != null) {
				callback.accept(img);
			}
			
		}
		
	}
	
}